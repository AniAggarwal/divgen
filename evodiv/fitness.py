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

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def render(self, latents: torch.Tensor, prompt: str) -> torch.Tensor:
        """Render a (N, C, H, W) latent tensor to (N, 3, Hpix, Wpix) in [0,1].

        Rendered in chunks of ``render_chunk`` with a fixed seeded generator so
        results are reproducible across a run.
        """
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
        torch.cuda.empty_cache()
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
    @torch.no_grad()
    def _evaluate_single(self, population, prompt: str, log_all: bool):
        """Evaluate on ONE prompt; return (F (P,2), raw list). No side effects."""
        spec = population.spec
        P, B = population.size, spec.set_size
        flat = population.latents.reshape(P * B, spec.channels, spec.height, spec.width)
        images = self.render(flat, prompt)               # (P*B, 3, Hpix, Wpix)
        images = images.reshape(P, B, *images.shape[1:])

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
    def evaluate(self, population, prompt, log_all: bool = False):
        """Evaluate every genome on ``prompt`` (str) or a list of prompts.

        For a list, fitness is the mean over prompts (island-model / multi-prompt
        generalisation). Sets ``population.F`` (P,2) = [-quality, -diversity] and
        ``population.raw`` (per-genome metric dicts).
        """
        if isinstance(prompt, str):
            F, raw = self._evaluate_single(population, prompt, log_all)
            population.F, population.raw = F, raw
            return
        # list of prompts: average
        Fs, raws = [], []
        for p in prompt:
            F, raw = self._evaluate_single(population, p, log_all)
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
