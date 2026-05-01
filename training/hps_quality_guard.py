import logging
from typing import Dict, List, Optional, Tuple

import torch

from rewards.base_reward import BaseRewardLoss


class HPSQualityGuard:
    """Per-sample HPS quality guard for latent optimization.

    Tracks per-sample HPS scores against per-sample thresholds and either
    freezes (monotonically) or reverts samples whose quality has degraded.
    Owns its state across iterations of a single train() call.

    `restored_mask` returned by `apply()` flags samples whose latent buffer
    is no longer the one the optimizer accumulated grads/momentum against —
    the caller must zero those samples' grad and optimizer state before the
    next step.
    """

    def __init__(
        self,
        latents: torch.Tensor,
        device: torch.device,
        reward_losses: List[BaseRewardLoss],
        sample_threshold: Optional[float],
        absolute_threshold: Optional[float],
        revert_mode: bool,
    ):
        self.device = device
        self.sample_threshold = sample_threshold
        self.absolute_threshold = absolute_threshold
        self.revert_mode = revert_mode

        self.num_samples = latents.shape[0]
        self.frozen_samples = torch.zeros(self.num_samples, dtype=torch.bool, device=device)
        self.last_good_latents = latents.detach().clone()
        self.revert_counts = torch.zeros(self.num_samples, dtype=torch.int32, device=device)
        self.per_sample_hps_thresholds: Optional[torch.Tensor] = None

        self.hps_reward_loss: Optional[BaseRewardLoss] = None
        if sample_threshold is not None:
            for rl in reward_losses:
                if rl.name == "HPS":
                    self.hps_reward_loss = rl
                    break
            if self.hps_reward_loss is None:
                logging.warning("hps_sample_threshold specified but HPS reward not enabled. Ignoring.")

    @property
    def enabled(self) -> bool:
        """True iff the guard should run this train() call."""
        return self.sample_threshold is not None and self.hps_reward_loss is not None

    def apply(
        self,
        preprocessed_image: torch.Tensor,
        prompt: str,
        latents: torch.Tensor,
    ) -> Tuple[bool, Optional[str], Dict[str, float], List[str], Optional[torch.Tensor]]:
        """Run the guard for one iteration.

        Computes per-sample HPS scores, lazily initializes per-sample
        thresholds on the first call, and dispatches to the freeze or revert
        branch. Mutates internal state and `latents.data` in place.

        Returns (should_stop, stop_reason, rewards_extras, log_fragments,
        restored_mask). `should_stop` is only ever True from the freeze path
        (when every sample is frozen).
        """
        with torch.no_grad():
            image_features = self.hps_reward_loss.get_image_features(preprocessed_image)
            text_features = self.hps_reward_loss.get_text_features(prompt)
            per_sample_scores = self.hps_reward_loss.get_per_sample_scores(image_features, text_features)

            if self.per_sample_hps_thresholds is None:
                initial_scores = per_sample_scores.detach().clone()
                if self.absolute_threshold is not None:
                    self.per_sample_hps_thresholds = torch.full_like(initial_scores, self.absolute_threshold)
                    logging.info(f"[HPS Freeze] Using absolute threshold: {self.absolute_threshold}")
                else:
                    self.per_sample_hps_thresholds = torch.minimum(
                        torch.full_like(initial_scores, self.sample_threshold),
                        initial_scores - 0.01,
                    )
                logging.info(f"[HPS Freeze] Initial scores: {initial_scores.tolist()}")
                logging.info(f"[HPS Freeze] Per-sample thresholds (stop when <=): {self.per_sample_hps_thresholds.tolist()}")

            below_threshold = per_sample_scores <= self.per_sample_hps_thresholds

            if self.revert_mode:
                rewards_extras, log_fragments, restored_mask = self._apply_revert(
                    below_threshold, per_sample_scores, latents
                )
                return False, None, rewards_extras, log_fragments, restored_mask
            return self._apply_freeze(below_threshold, per_sample_scores, latents)

    def _apply_revert(
        self,
        below_threshold: torch.Tensor,
        per_sample_scores: torch.Tensor,
        latents: torch.Tensor,
    ) -> Tuple[Dict[str, float], List[str], Optional[torch.Tensor]]:
        """For each sample below threshold: restore latents.data[idx] to
        last_good_latents[idx] and increment revert_counts[idx]. For samples
        above threshold: refresh last_good_latents. Never stops.

        The returned restored_mask is the per-iteration `below_threshold`
        set whose grad+optimizer state must be zeroed before the next step
        (else Adam/SGD momentum from the divergent step keeps moving the
        just-restored latent in the bad direction).
        """
        if below_threshold.any():
            revert_indices = torch.where(below_threshold)[0].tolist()
            for idx in revert_indices:
                logging.info(
                    f"[HPS Revert] Sample {idx} quality degraded below threshold "
                    f"{self.per_sample_hps_thresholds[idx].item():.4f} "
                    f"(score: {per_sample_scores[idx].item():.4f})"
                )
                latents.data[idx] = self.last_good_latents[idx].clone()
                self.revert_counts[idx] += 1
                logging.info(
                    f"[HPS Revert] Reverted sample {idx} to last good latents "
                    f"(revert #{self.revert_counts[idx].item()})"
                )

        good_samples = ~below_threshold
        if good_samples.any():
            for idx in torch.where(good_samples)[0]:
                self.last_good_latents[idx] = latents.data[idx].detach().clone()

        total_reverts = self.revert_counts.sum().item()
        restored_mask = below_threshold if below_threshold.any() else None
        return ({"total_reverts": total_reverts}, [f"[Reverts: {total_reverts}]"], restored_mask)

    def _apply_freeze(
        self,
        below_threshold: torch.Tensor,
        per_sample_scores: torch.Tensor,
        latents: torch.Tensor,
    ) -> Tuple[bool, Optional[str], Dict[str, float], List[str], Optional[torch.Tensor]]:
        """Newly-frozen samples (below threshold this iter, not previously
        frozen) get their latents restored to the last known good state.
        The frozen mask is monotonically growing. Triggers should_stop when
        every sample is frozen.

        The returned restored_mask is the cumulative `frozen_samples` set:
        those latents must keep their grad+optimizer state zeroed every
        iteration so they truly stop moving (otherwise momentum keeps
        accumulating from non-zero gradients on the live forward pass).
        """
        just_frozen = below_threshold & ~self.frozen_samples
        if just_frozen.any():
            frozen_indices = torch.where(just_frozen)[0].tolist()
            for idx in frozen_indices:
                logging.info(
                    f"[HPS Freeze] Sample {idx} quality degraded below threshold "
                    f"{self.per_sample_hps_thresholds[idx].item():.4f} "
                    f"(score: {per_sample_scores[idx].item():.4f})"
                )
                latents.data[idx] = self.last_good_latents[idx].clone()
                logging.info(f"[HPS Freeze] Restored sample {idx} to last good latents")
        self.frozen_samples = self.frozen_samples | below_threshold

        good_samples = ~self.frozen_samples
        if good_samples.any():
            for idx in torch.where(good_samples)[0]:
                self.last_good_latents[idx] = latents.data[idx].detach().clone()

        num_frozen = self.frozen_samples.sum().item()
        rewards_extras = {"frozen_samples": num_frozen}
        log_fragments = [f"[Frozen: {num_frozen}/{self.num_samples}]"]

        should_stop = bool(self.frozen_samples.all())
        stop_reason = None
        if should_stop:
            stop_reason = f"All {self.num_samples} samples degraded below HPS thresholds"
            logging.info(f"[HPS Freeze] {stop_reason}")

        restored_mask = self.frozen_samples if self.frozen_samples.any() else None
        return should_stop, stop_reason, rewards_extras, log_fragments, restored_mask
