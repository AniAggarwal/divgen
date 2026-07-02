"""Fitness evaluation for EvoDiv noise-set genomes.

A genome's fitness has two objectives, both reusing the *base method's own*
scoring code so the numbers are directly comparable to the paper:

  * quality   -- mean per-image reward over the set (CLIPScore or HPSv2), the
                 same rewards used in Tab. 1 / Tab. 2. NOT ImageReward.
  * diversity -- a set-level diversity statistic (DINOv2 pairwise, DPP, or
                 Vendi) from ``objectives/``.

The whole population shares one prompt, so every genome's latents are rendered
in a *single* batched forward pass (chunked to respect a memory cap). This is
the key efficiency win over gradient-based noise optimisation: no backprop, no
per-sample optimiser state, just forward diffusion.

Objectives are returned in pymoo's minimisation convention as ``F = [-quality,
-diversity]`` so lower is better on both axes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from rewards import clip_img_transform


class FitnessEvaluator:
    def __init__(
        self,
        model,
        reward_losses: list,
        diversity_objectives: list,
        eval_diversity_objectives: Optional[list] = None,
        eval_reward_losses: Optional[list] = None,
        n_inference_steps: int = 1,
        seed: int = 0,
        device=torch.device("cuda"),
        model_dtype: torch.dtype = torch.float16,
        render_chunk: int = 128,
        multi_apply_fn=None,
    ):
        """
        Args:
            model: a divgen Reward*Pipeline with ``.apply(...)``.
            reward_losses: quality rewards that DEFINE the quality objective
                (their per-image scores are averaged, then summed with weights).
            diversity_objectives: objectives that DEFINE the diversity objective
                (summed with weights; the raw score of the first is the headline).
            eval_diversity_objectives / eval_reward_losses: extra, non-optimised
                metrics computed for logging only (held-out generalisation).
            render_chunk: max images per forward pass (memory cap).
            multi_apply_fn: optional multi-step renderer (flux-schnell) used only
                for the final re-render of selected genomes.
        """
        self.model = model
        self.reward_losses = reward_losses
        self.diversity_objectives = diversity_objectives
        self.eval_diversity_objectives = eval_diversity_objectives or []
        self.eval_reward_losses = eval_reward_losses or []
        self.n_inference_steps = n_inference_steps
        self.seed = seed
        self.device = device
        self.model_dtype = model_dtype
        self.render_chunk = render_chunk
        self.multi_apply_fn = multi_apply_fn
        self.preprocess = clip_img_transform(224)
        self.n_renders = 0   # running count of images rendered (compute accounting)

        # --- B200 fast paths (both verified equivalent; see evodiv/README) --- #
        # surrogate decoder (TAESD): ~10x cheaper VAE decode used for *search*
        # generations only; gen-0, periodic re-anchoring and final selection all
        # use the exact full VAE so reported numbers are unaffected.
        self.surrogate_decoder = None
        # batched scoring: one DINO / CLIP forward for the whole population
        # instead of one per genome (numerically identical math).
        self.batched_scoring = True

    def enable_surrogate(self, cache_dir: str):
        """Load TAESD (madebyollin/taesdxl) as the search-time decoder."""
        from diffusers import AutoencoderTiny
        self.surrogate_decoder = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=self.model_dtype, cache_dir=cache_dir,
        ).to(self.device).eval()

    def _surrogate_decode(self, latents: torch.Tensor) -> torch.Tensor:
        # TAESD consumes the scheduler's (scaled) latents directly.
        img = self.surrogate_decoder.decode(latents).sample
        return (img / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def render(self, latents: torch.Tensor, prompt: str, exact: bool = True) -> torch.Tensor:
        """Render a (N, C, H, W) latent tensor to (N, 3, Hpix, Wpix) in [0,1].

        Rendered in chunks of ``render_chunk`` with a fixed seeded generator so
        results are reproducible across a run. With ``exact=False`` and a loaded
        surrogate decoder, the final VAE decode is swapped for TAESD (~10x
        faster; diversity rank correlation 0.97 vs the full VAE).
        """
        use_surrogate = (not exact) and (self.surrogate_decoder is not None)
        orig_decode = None
        if use_surrogate and hasattr(self.model, "decode_latents_tensors"):
            orig_decode = self.model.decode_latents_tensors
            self.model.decode_latents_tensors = self._surrogate_decode
        try:
            n = latents.shape[0]
            out: List[torch.Tensor] = []
            for start in range(0, n, self.render_chunk):
                chunk = latents[start:start + self.render_chunk].to(self.model_dtype)
                gen = torch.Generator("cuda").manual_seed(self.seed)
                img = self.model.apply(
                    latents=chunk,
                    prompt=prompt,
                    generator=gen,
                    num_inference_steps=self.n_inference_steps,
                    num_images_per_prompt=chunk.shape[0],
                )
                out.append(img.float())
                self.n_renders += chunk.shape[0]
                del img, gen
        finally:
            if orig_decode is not None:
                self.model.decode_latents_tensors = orig_decode
        return torch.cat(out, 0)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _quality_of_set(self, images: torch.Tensor, prompt: str, rewards: list) -> Dict[str, float]:
        """Mean per-image reward scores for one set of images."""
        prep = self.preprocess(images)
        scores: Dict[str, float] = {}
        for rl in rewards:
            loss = rl(prep, prompt)
            if rl.name == "HPS":
                scores[rl.name] = 1 - loss.item()
            elif rl.name in ("CLIP", "CLIP-B32"):
                scores[rl.name] = 1 - loss.item() / 100
            else:
                scores[rl.name] = loss.item()
        return scores

    @torch.no_grad()
    def _diversity_of_set(self, images: torch.Tensor, objs: list) -> Dict[str, float]:
        """Set-level diversity scores, HIGHER = MORE DIVERSE for every objective.

        ``compute_pairwise_diversity`` already returns a higher-is-better score
        for all divgen objectives (pairwise distance for DINO/LPIPS/DreamSim/
        color/tiny-L2, the Vendi score for Vendi, and a normalised log-det for
        DPP/DPP-patch). We use it directly as the maximisation signal; for DPP we
        additionally surface the interpretable raw log-det under ``<name>_raw``.
        """
        scores: Dict[str, float] = {}
        for obj in objs:
            d = obj.compute_pairwise_diversity(images)   # higher = better; also sets _last_raw_score for DPP
            scores[obj.name] = float(d.item()) if torch.is_tensor(d) else float(d)
            raw = getattr(obj, "_last_raw_score", None)
            if raw is not None and obj.name in ("diversity_dpp", "diversity_dpp_patch"):
                scores[f"{obj.name}_raw"] = float(raw)
        return scores

    # ------------------------------------------------------------------ #
    def _can_batch(self, log_all: bool) -> bool:
        """Batched fast path applies to the (CLIP, diversity_dino) objective pair."""
        return (
            self.batched_scoring and not log_all
            and len(self.reward_losses) == 1 and self.reward_losses[0].name == "CLIP"
            and len(self.diversity_objectives) == 1
            and self.diversity_objectives[0].name == "diversity_dino"
        )

    @torch.no_grad()
    def _score_population_batched(self, images: torch.Tensor, prompt: str, P: int, B: int):
        """One DINO + one CLIP forward for the whole population.

        Numerically equivalent to the per-set loop (verified: max abs diff
        <1e-3 on both metrics); just batches the feature extraction.
        Returns (F (P,2), raw list).
        """
        import torch.nn.functional as NF
        flat_imgs = images.reshape(P * B, *images.shape[2:])

        # --- CLIP quality: score per image, then mean per set ---------------- #
        rl = self.reward_losses[0]
        prep = self.preprocess(flat_imgs)
        img_feats = rl.get_image_features(prep)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        if prompt not in rl._text_feature_cache:
            rl._text_feature_cache[prompt] = rl.get_text_features(prompt)
        txt = rl._text_feature_cache[prompt]
        txt = txt / txt.norm(dim=-1, keepdim=True)
        scale = rl.clip_model.logit_scale.exp()
        sim = (img_feats @ txt.T).squeeze(-1) * scale          # (P*B,)
        quality = (sim.reshape(P, B).mean(dim=1) / 100.0).float()   # == 1 - loss/100

        # --- DINO patch diversity: batched pairwise cosine ------------------- #
        obj = self.diversity_objectives[0]
        feats = obj.dino_backend.extract_features(flat_imgs, normalize=False, use_cls=False)
        feats = NF.normalize(feats, p=2, dim=-1)               # (P*B, N, D)
        feats = feats.reshape(P, B, *feats.shape[1:])          # (P, B, N, D)
        sim_m = torch.einsum("pind,pjnd->pijn", feats, feats)  # cosine (normalized)
        dist = (1.0 - sim_m).mean(dim=-1)                      # (P, B, B)
        iu, ju = torch.triu_indices(B, B, offset=1)
        diversity = dist[:, iu, ju].mean(dim=1).float()        # (P,)

        F = torch.stack([-rl.weighting * quality, -obj.weighting * diversity], dim=1)
        raw = [
            {"CLIP": float(quality[i]), "diversity_dino": float(diversity[i]),
             "_quality": float(rl.weighting * quality[i]),
             "_diversity": float(obj.weighting * diversity[i])}
            for i in range(P)
        ]
        return F.to(self.device), raw

    @torch.no_grad()
    def _evaluate_single(self, population, prompt: str, log_all: bool, exact: bool = True):
        """Evaluate on ONE prompt; return (F (P,2), raw list). No side effects."""
        spec = population.spec
        P, B = population.size, spec.set_size
        flat = population.latents.reshape(P * B, spec.channels, spec.height, spec.width)
        images = self.render(flat, prompt, exact=exact)  # (P*B, 3, Hpix, Wpix)
        images = images.reshape(P, B, *images.shape[1:])

        if self._can_batch(log_all):
            F, raw = self._score_population_batched(images, prompt, P, B)
            del images, flat
            return F, raw

        F = torch.zeros(P, 2, device=self.device)
        raw: List[Dict[str, float]] = []
        for i in range(P):
            imgs = images[i]
            q_scores = self._quality_of_set(imgs, prompt, self.reward_losses)
            d_scores = self._diversity_of_set(imgs, self.diversity_objectives)
            quality = sum(rl.weighting * q_scores[rl.name] for rl in self.reward_losses)
            diversity = sum(o.weighting * d_scores[o.name] for o in self.diversity_objectives)
            rec = dict(q_scores)
            rec.update(d_scores)
            if log_all:
                if self.eval_reward_losses:
                    rec.update(self._quality_of_set(imgs, prompt, self.eval_reward_losses))
                if self.eval_diversity_objectives:
                    rec.update(self._diversity_of_set(imgs, self.eval_diversity_objectives))
            rec["_quality"] = float(quality)
            rec["_diversity"] = float(diversity)
            raw.append(rec)
            F[i, 0] = -quality
            F[i, 1] = -diversity
        del images, flat
        torch.cuda.empty_cache()
        return F, raw

    @torch.no_grad()
    def evaluate(self, population, prompt, log_all: bool = False, exact: bool = True):
        """Evaluate every genome on ``prompt`` (str) or a list of prompts.

        For a list, fitness is the mean over prompts (island-model / multi-prompt
        generalisation). Sets ``population.F`` (P,2) = [-quality, -diversity] and
        ``population.raw`` (per-genome metric dicts). ``exact=False`` uses the
        TAESD surrogate decoder (search-time only) when one is loaded.
        """
        if isinstance(prompt, str):
            F, raw = self._evaluate_single(population, prompt, log_all, exact=exact)
            population.F, population.raw = F, raw
            return
        # list of prompts: average
        Fs, raws = [], []
        for p in prompt:
            F, raw = self._evaluate_single(population, p, log_all, exact=exact)
            Fs.append(F)
            raws.append(raw)
        population.F = torch.stack(Fs, 0).mean(0)
        merged: List[Dict[str, float]] = []
        for i in range(population.size):
            acc: Dict[str, float] = {}
            for r in raws:
                for k, v in r[i].items():
                    acc[k] = acc.get(k, 0.0) + v / len(prompt)
            merged.append(acc)
        population.raw = merged
