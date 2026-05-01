import logging
import math
from typing import Callable, Dict, List, Optional, Set, Tuple

import PIL
import PIL.Image
import torch
from diffusers import DiffusionPipeline

from rewards import clip_img_transform
from rewards.base_reward import BaseRewardLoss
from objectives.base_objective import BaseDiversityObjective
from training.early_stopping import (
    calculate_dynamic_thresholds,
    check_early_stopping,
    log_early_stopping,
)
from training.hps_quality_guard import HPSQualityGuard


class LatentNoiseTrainer:
    """Trainer for optimizing latents with reward losses."""

    def __init__(
        self,
        reward_losses: List[BaseRewardLoss],
        model: DiffusionPipeline,
        n_iters: int,
        n_inference_steps: int,
        seed: int,
        regularize: bool = True,
        regularization_weight: float = 0.01,
        grad_clip: float = 0.1,
        log_metrics: bool = True,
        device: torch.device = torch.device("cuda"),
        diversity_objectives: Optional[List[BaseDiversityObjective]] = None,
        diversity_thresholds: Optional[Dict[str, float]] = None,
        all_reward_losses: Optional[List[BaseRewardLoss]] = None,
        all_diversity_objectives: Optional[List[BaseDiversityObjective]] = None,
    ):
        self.reward_losses = reward_losses
        self.diversity_objectives = diversity_objectives if diversity_objectives is not None else []
        # Comprehensive metrics for logging (all available metrics, not just enabled ones)
        self.all_reward_losses = all_reward_losses if all_reward_losses is not None else reward_losses
        self.all_diversity_objectives = all_diversity_objectives if all_diversity_objectives is not None else self.diversity_objectives
        self.model = model
        self.n_iters = n_iters
        self.n_inference_steps = n_inference_steps
        self.seed = seed
        self.regularize = regularize
        self.regularization_weight = regularization_weight
        self.grad_clip = grad_clip
        self.log_metrics = log_metrics
        self.device = device
        self.preprocess_fn = clip_img_transform(224)
        self.diversity_thresholds = diversity_thresholds if diversity_thresholds is not None else {}
        self._base_diversity_objectives = list(self.diversity_objectives)
        self.diversity_reference_images: Optional[torch.Tensor] = None
        self.diversity_reference_objectives: Optional[Set[str]] = None

    @staticmethod
    def _tensor_to_pil(batch: torch.Tensor) -> List[PIL.Image.Image]:
        image_numpy = batch.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return DiffusionPipeline.numpy_to_pil(image_numpy)

    def _compute_all_metrics(self, image: torch.Tensor, prompt: str, load_all_metrics: bool = True) -> Dict[str, float]:
        """Compute all reward and diversity metrics for a given image.

        This computes ALL available metrics (not just enabled ones) without gradients
        for comprehensive logging purposes.

        Args:
            image: Image tensor to compute metrics for
            prompt: Text prompt associated with the image
            load_all_metrics: If True, temporarily load all metrics for comprehensive logging

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}

        # Compute without gradients for logging
        with torch.no_grad():
            preprocessed_image = self.preprocess_fn(image)

            # Compute ALL reward losses (including disabled ones)
            for reward_loss in self.all_reward_losses:
                try:
                    loss = reward_loss(preprocessed_image, prompt)
                    if reward_loss.name == "HPS":
                        metrics[reward_loss.name] = 1 - loss.item()
                    elif reward_loss.name == "CLIP":
                        metrics[reward_loss.name] = 1 - loss.item() / 100
                    else:
                        metrics[reward_loss.name] = loss.item()
                except Exception as e:
                    logging.warning(f"Failed to compute {reward_loss.name}: {e}")
                    metrics[reward_loss.name] = float('nan')

            # Compute ALL diversity objectives (including disabled ones)
            for diversity_obj in self.all_diversity_objectives:
                try:
                    diversity_loss = diversity_obj(image)
                    if hasattr(diversity_obj, '_last_raw_score'):
                        metrics[diversity_obj.name] = diversity_obj._last_raw_score
                    else:
                        metrics[diversity_obj.name] = -diversity_loss.item()
                except Exception as e:
                    logging.warning(f"Failed to compute {diversity_obj.name}: {e}")
                    metrics[diversity_obj.name] = float('nan')

            # If we loaded extra metrics, compute them and then clean up
            if load_all_metrics and hasattr(self, '_temp_logging_objectives'):
                for diversity_obj in self._temp_logging_objectives:
                    try:
                        diversity_loss = diversity_obj(image)
                        if hasattr(diversity_obj, '_last_raw_score'):
                            metrics[diversity_obj.name] = diversity_obj._last_raw_score
                        else:
                            metrics[diversity_obj.name] = -diversity_loss.item()
                    except Exception as e:
                        logging.warning(f"Failed to compute {diversity_obj.name}: {e}")
                        metrics[diversity_obj.name] = float('nan')

            # Delete preprocessed_image to free memory
            del preprocessed_image

        return metrics

    # ------------------------------------------------------------------ #
    # train() helpers — pure-as-possible extractions of train() phases.  #
    # ------------------------------------------------------------------ #

    def _generate_image(self, current_latents: torch.Tensor, prompt: str) -> torch.Tensor:
        """Run the diffusion pipeline forward with a fresh seeded generator.

        Calls torch.cuda.empty_cache() internally to release intermediate
        activations, matching the original train() behavior.
        """
        generator = torch.Generator("cuda").manual_seed(self.seed)
        image = self.model.apply(
            latents=current_latents,
            prompt=prompt,
            generator=generator,
            num_inference_steps=self.n_inference_steps,
            num_images_per_prompt=current_latents.shape[0],
        )
        del generator
        torch.cuda.empty_cache()
        return image

    def _capture_initial(
        self,
        image: torch.Tensor,
        current_latents: torch.Tensor,
        prompt: str,
        multi_apply_fn,
    ) -> Tuple[List[PIL.Image.Image], Dict[str, float]]:
        """Compute the iteration-0 'initial' images and their full metric set.

        If multi_apply_fn is provided it is used to render the initial batch
        from the latents (multi-step rendering); otherwise the just-generated
        single-step image is reused.
        """
        if multi_apply_fn is not None:
            initial_batch = multi_apply_fn(current_latents.detach(), prompt)
        else:
            initial_batch = image.detach()
        initial_images = self._tensor_to_pil(initial_batch)
        initial_rewards = self._compute_all_metrics(initial_batch, prompt)
        logging.info(f"Initial metrics: {initial_rewards}")
        del initial_batch
        torch.cuda.empty_cache()
        return initial_images, initial_rewards

    def _compute_reward_losses(
        self,
        image: torch.Tensor,
        prompt: str,
        hps_in_warmup: bool,
        total_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[str], torch.Tensor]:
        """Run all reward losses, accumulate into total_loss, build log fragments.

        HPS during warmup is logged with `(warmup)` suffix on the same fragment
        and is NOT added to total_loss.

        Returns the updated total_loss, a per-reward score dict, log fragments,
        and the preprocessed image tensor (so the HPS quality guard can reuse
        it without preprocessing twice).
        """
        rewards: Dict[str, float] = {}
        log_fragments: List[str] = []
        preprocessed_image = self.preprocess_fn(image)
        for reward_loss in self.reward_losses:
            loss = reward_loss(preprocessed_image, prompt).to(total_loss.dtype)
            if reward_loss.name == "HPS":
                score = 1 - loss.item()
            elif reward_loss.name == "CLIP":
                score = 1 - loss.item() / 100
            else:
                score = loss.item()
            frag = f"{reward_loss.name}: {score:.4f}"
            if reward_loss.name == "HPS" and hps_in_warmup:
                frag += "(warmup)"
            else:
                total_loss = total_loss + loss * reward_loss.weighting
            log_fragments.append(frag)
            rewards[reward_loss.name] = score
        return total_loss, rewards, log_fragments, preprocessed_image

    def _compute_diversity_losses(
        self,
        image: torch.Tensor,
        total_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[str]]:
        """Run all diversity objectives, accumulate into total_loss.

        Honors `self.diversity_reference_images` and the per-objective
        membership in `self.diversity_reference_objectives` so that sequential
        rounds prepend reference images to the diversity input.
        """
        rewards: Dict[str, float] = {}
        log_fragments: List[str] = []
        for diversity_obj in self.diversity_objectives:
            diversity_input = image
            use_references = (
                self.diversity_reference_images is not None
                and (
                    self.diversity_reference_objectives is None
                    or diversity_obj.name in self.diversity_reference_objectives
                )
            )
            if use_references:
                reference_batch = self.diversity_reference_images.to(
                    image.device, dtype=image.dtype
                )
                reference_batch = reference_batch.detach()
                diversity_input = torch.cat([reference_batch, image], dim=0)
            diversity_loss = diversity_obj(diversity_input).to(total_loss.dtype)
            if hasattr(diversity_obj, '_last_raw_score'):
                rewards[diversity_obj.name] = diversity_obj._last_raw_score
            else:
                rewards[diversity_obj.name] = -diversity_loss.item()
            log_fragments.append(f"{diversity_obj.name}: {-diversity_loss.item():.4f}")
            total_loss = total_loss + diversity_obj.weighting * diversity_loss
        return total_loss, rewards, log_fragments

    def _init_dynamic_thresholds_if_first(
        self,
        iteration: int,
        rewards: Dict[str, float],
        dynamic_thresholds_calculated: Dict,
        initial_diversity_values: Dict[str, float],
    ) -> None:
        """On iteration 0, compute dynamic thresholds and snapshot initial
        diversity values. Mutates the two accumulator dicts in place.
        """
        if iteration != 0:
            return
        calculated = calculate_dynamic_thresholds(
            rewards, self.diversity_thresholds, iteration
        )
        dynamic_thresholds_calculated.update(calculated)
        for metric_name in calculated.keys():
            if metric_name in rewards:
                initial_diversity_values[metric_name] = rewards[metric_name]

    def _check_stops(
        self,
        rewards: Dict[str, float],
        iteration: int,
        dynamic_thresholds_calculated: Dict,
        initial_diversity_values: Dict[str, float],
        extra_stop_fn: Optional[Callable[[Dict[str, float], int], Optional[str]]],
        should_stop: bool,
        stop_reason: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Combine built-in early-stopping checks with the optional
        extra_stop_fn callback. Preserves the original semantics: if HPS
        already requested a stop, that wins; built-in early stops fire next;
        the custom callback can override only if neither has fired.
        """
        if not should_stop:
            should_stop, stop_reason = check_early_stopping(
                rewards,
                self.diversity_thresholds,
                dynamic_thresholds_calculated,
                initial_diversity_values,
            )
        if not should_stop and extra_stop_fn is not None:
            try:
                custom_stop_reason = extra_stop_fn(rewards, iteration)
            except Exception as extra_exc:
                logging.warning(f"Custom stop function raised an error: {extra_exc}")
                custom_stop_reason = None
            if custom_stop_reason:
                should_stop = True
                stop_reason = custom_stop_reason
                logging.info(f"[Custom Stop] {custom_stop_reason}")
        return should_stop, stop_reason

    def _apply_regularization(
        self,
        current_latents: torch.Tensor,
        total_loss: torch.Tensor,
        rewards: Dict[str, float],
        latent_dim: int,
    ) -> torch.Tensor:
        """Add the spherical-boundary regularization term to total_loss.

        Computed in fp32 to avoid overflow. Mutates `rewards` to add the
        "norm" entry (mean per-sample latent L2 norm).
        """
        flat_latents = current_latents.view(current_latents.shape[0], -1)
        latent_norm = torch.linalg.vector_norm(flat_latents, dim=1).to(torch.float32)
        safe_norm = torch.clamp(latent_norm, min=1e-6)
        regularization_per_sample = (
            0.5 * safe_norm**2 - (latent_dim - 1) * torch.log(safe_norm)
        )
        regularization = self.regularization_weight * regularization_per_sample.mean()
        rewards["norm"] = latent_norm.mean().item()
        total_loss = total_loss + regularization.to(total_loss.dtype)
        del flat_latents, latent_norm, safe_norm, regularization_per_sample, regularization
        return total_loss

    def _record_iteration(
        self,
        iteration: int,
        rewards: Dict[str, float],
        total_loss: torch.Tensor,
        return_history: bool,
        iteration_history: Optional[List[Dict[str, float]]],
    ) -> None:
        """Append per-iteration metrics to history. No-op if not recording."""
        if not return_history:
            return
        iter_metrics = {
            "iteration": iteration,
            "loss/total": total_loss.item(),
        }
        iter_metrics.update({f"reward/{key}": value for key, value in rewards.items()})
        iteration_history.append(iter_metrics)

    @staticmethod
    def _update_best(
        image: torch.Tensor,
        latents: torch.Tensor,
        rewards: Dict[str, float],
        total_reward_loss: float,
        best_loss: float,
        best_image: Optional[torch.Tensor],
        best_rewards: Optional[Dict[str, float]],
        best_latents: Optional[torch.Tensor],
    ) -> Tuple[float, Optional[torch.Tensor], Optional[Dict[str, float]], Optional[torch.Tensor], bool]:
        """If the current iteration improved on best_loss, snapshot the image,
        rewards, and latents. Must be called BEFORE backward()."""
        is_best_image = total_reward_loss < best_loss
        if is_best_image:
            best_loss = total_reward_loss
            best_image = image.detach().clone() if hasattr(image, 'detach') else image
            best_rewards = rewards
            best_latents = latents.detach().cpu()
        return best_loss, best_image, best_rewards, best_latents, is_best_image

    @staticmethod
    def _zero_optimizer_state(
        latents: torch.Tensor,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Zero gradients and optimizer momentum/variance buffers along the
        leading (sample) dimension of `latents` for the indices flagged in
        `mask`.

        Required for freeze and revert modes: when `latents.data[idx]` has
        been overwritten from a snapshot, the gradient just produced by
        backward() and any Adam/SGD momentum carrying state from the
        divergent step would otherwise immediately move the restored latent
        in the wrong direction on the next optimizer.step().

        Covers SGD-with-momentum (`momentum_buffer`) and Adam variants
        (`exp_avg`, `exp_avg_sq`). Other state keys are left alone.
        """
        if mask is None or not mask.any():
            return
        idxs = torch.where(mask)[0]
        if latents.grad is not None:
            for idx in idxs:
                latents.grad.data[idx] = 0.0
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p is latents and p in optimizer.state:
                    state = optimizer.state[p]
                    for key in ('momentum_buffer', 'exp_avg', 'exp_avg_sq'):
                        buf = state.get(key)
                        if buf is not None:
                            for idx in idxs:
                                buf[idx] = 0.0

    def _step_optimizer(
        self,
        total_loss: torch.Tensor,
        latents: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        image: torch.Tensor,
        is_best_image: bool,
        iteration: int,
        restored_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Run backward + (optional) per-reward CPU offload + grad clip +
        zero-out for restored samples + optimizer step. Cleans up `image`
        and `total_loss` and empties the cache.

        `restored_mask` flags samples whose latent buffer was overwritten by
        the HPS quality guard this iteration; their gradient and optimizer
        momentum/variance are zeroed before stepping so the restoration
        actually sticks.
        """
        total_loss.backward()

        # After backward pass, offload reward models to CPU if using lazy loading
        for reward_loss in self.reward_losses:
            if hasattr(reward_loss, 'offload_after_backward'):
                reward_loss.offload_after_backward()

        torch.nn.utils.clip_grad_norm_(latents, self.grad_clip)
        self._zero_optimizer_state(latents, restored_mask, optimizer)
        optimizer.step()

        # Clear memory after optimizer step - delete image and other large tensors.
        # Note: best_image was already cloned above; safe to delete the working copy.
        if iteration != self.n_iters - 1 and not is_best_image:
            del image
        del total_loss
        torch.cuda.empty_cache()

    def _finalize_best(
        self,
        best_image: torch.Tensor,
        best_latents: Optional[torch.Tensor],
        best_rewards: Optional[Dict[str, float]],
        prompt: str,
        multi_apply_fn,
        skip_final_metrics_log: bool,
    ) -> Tuple[List[PIL.Image.Image], Dict[str, float]]:
        """Convert the best tensor to PIL and recompute its metrics. If a
        multi-step renderer is provided, re-render the best latents with it
        and use that image for the final metrics.
        """
        best_image_pil = self._tensor_to_pil(best_image)
        if multi_apply_fn is not None and best_latents is not None:
            multi_step_image = multi_apply_fn(best_latents.to("cuda"), prompt)
            best_image_pil = self._tensor_to_pil(multi_step_image)
            best_rewards = self._compute_all_metrics(multi_step_image, prompt)
            if not skip_final_metrics_log:
                logging.info(f"Best metrics (after multi-step): {best_rewards}")
            del multi_step_image
        else:
            best_rewards = self._compute_all_metrics(best_image, prompt)
            if not skip_final_metrics_log:
                logging.info(f"Best metrics: {best_rewards}")
        return best_image_pil, best_rewards

    # ------------------------------------------------------------------ #
    # Public training loop. Reads top-to-bottom as a sequence of phases. #
    # ------------------------------------------------------------------ #

    def train(
        self,
        latents: torch.Tensor,
        prompt: str,
        optimizer: torch.optim.Optimizer,
        save_dir: Optional[str] = None,
        multi_apply_fn=None,
        return_history: bool = False,
        extra_stop_fn: Optional[Callable[[Dict[str, float], int], Optional[str]]] = None,
        skip_final_metrics_log: bool = False,
        hps_sample_threshold: Optional[float] = None,
        hps_absolute_threshold: Optional[float] = None,
        hps_revert_mode: bool = False,
        hps_warmup_iters: int = 0,
    ) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image], Dict[str, float], Dict[str, float], Optional[List[Dict[str, float]]]]:
        logging.info(f"Optimizing latents for prompt '{prompt}'.")
        best_loss = torch.inf
        best_image = best_latents = best_rewards = None
        initial_images = initial_rewards = None
        iteration_history = [] if return_history else None
        initial_diversity_values: Dict[str, float] = {}
        dynamic_thresholds_calculated: Dict = {}
        use_dynamic = self.diversity_thresholds.get('use_dynamic_thresholds', False)
        default_multiplier = self.diversity_thresholds.get('dynamic_threshold_multiplier', 4.0)
        latent_dim = math.prod(latents.shape[1:])

        hps_guard = HPSQualityGuard(
            latents=latents,
            device=self.device,
            reward_losses=self.reward_losses,
            sample_threshold=hps_sample_threshold,
            absolute_threshold=hps_absolute_threshold,
            revert_mode=hps_revert_mode,
        )

        for iteration in range(self.n_iters):
            log_fragments: List[str] = []
            rewards: Dict[str, float] = {}
            should_stop = False
            stop_reason: Optional[str] = None
            restored_mask: Optional[torch.Tensor] = None
            optimizer.zero_grad()

            current_latents = latents
            image = self._generate_image(current_latents, prompt)

            if initial_images is None:
                initial_images, initial_rewards = self._capture_initial(
                    image, current_latents, prompt, multi_apply_fn
                )

            total_loss = torch.zeros((), device=image.device, dtype=latents.dtype, requires_grad=False)
            hps_in_warmup = iteration < hps_warmup_iters
            total_loss, r, frags, preprocessed_image = self._compute_reward_losses(
                image, prompt, hps_in_warmup, total_loss
            )
            rewards.update(r)
            log_fragments.extend(frags)

            if hps_guard.enabled:
                should_stop, stop_reason, r, frags, restored_mask = hps_guard.apply(
                    preprocessed_image, prompt, latents,
                )
                rewards.update(r)
                log_fragments.extend(frags)

            total_loss, r, frags = self._compute_diversity_losses(image, total_loss)
            rewards.update(r)
            log_fragments.extend(frags)

            self._init_dynamic_thresholds_if_first(
                iteration, rewards, dynamic_thresholds_calculated, initial_diversity_values
            )
            should_stop, stop_reason = self._check_stops(
                rewards, iteration, dynamic_thresholds_calculated,
                initial_diversity_values, extra_stop_fn, should_stop, stop_reason,
            )

            rewards["total"] = total_loss.item()
            total_reward_loss = total_loss.item()
            if self.regularize:
                total_loss = self._apply_regularization(current_latents, total_loss, rewards, latent_dim)

            if self.log_metrics:
                logging.info(f"Iteration {iteration}: {', '.join(log_fragments)}")
            self._record_iteration(iteration, rewards, total_loss, return_history, iteration_history)

            best_loss, best_image, best_rewards, best_latents, is_best_image = self._update_best(
                image, latents, rewards, total_reward_loss,
                best_loss, best_image, best_rewards, best_latents,
            )

            if should_stop:
                log_early_stopping(iteration, stop_reason)
                if best_image is None:
                    best_image = image.detach().clone() if hasattr(image, 'detach') else image
                    best_rewards = rewards
                    best_latents = latents.detach().cpu()
                break

            if iteration != self.n_iters - 1:
                self._step_optimizer(
                    total_loss, latents, optimizer, image, is_best_image, iteration,
                    restored_mask=restored_mask,
                )

        best_image_pil, best_rewards = self._finalize_best(
            best_image, best_latents, best_rewards, prompt, multi_apply_fn, skip_final_metrics_log,
        )
        del best_image, best_latents
        if 'image' in locals():
            del image
        if 'latents' in locals():
            del latents
        torch.cuda.empty_cache()

        return initial_images, best_image_pil, initial_rewards, best_rewards, iteration_history
