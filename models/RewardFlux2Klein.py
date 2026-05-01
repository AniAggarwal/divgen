import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import Flux2KleinPipeline

from models.utils import freeze_params


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values."
        )

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom "
                "timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom "
                "sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class RewardFlux2KleinPipeline(Flux2KleinPipeline):
    """Differentiable FLUX.2-klein wrapper for latent optimization."""

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        transformer,
        is_distilled: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
            is_distilled=is_distilled,
        )

        if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
            self.text_encoder.gradient_checkpointing_enable()
        if hasattr(self.text_encoder, "enable_gradient_checkpointing"):
            self.text_encoder.enable_gradient_checkpointing()
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()
        if hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()

        self.vae.eval()
        self.text_encoder.eval()
        self.transformer.eval()

        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())
        freeze_params(self.transformer.parameters())

    def apply(
        self,
        latents: Optional[torch.FloatTensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        max_sequence_length: int = 512,
        output_type: str = "pt",
        use_activation_checkpointing: bool = True,
        timestep_checkpointing_interval: Optional[int] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]

        if num_inference_steps < 1:
            raise ValueError(
                f"`num_inference_steps` must be >= 1, got {num_inference_steps}."
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(
            image_seq_len=image_seq_len,
            num_steps=num_inference_steps,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        if timestep_checkpointing_interval is None:
            timestep_checkpointing_interval = len(timesteps)
        if timestep_checkpointing_interval < 1:
            raise ValueError(
                "`timestep_checkpointing_interval` must be >= 1, "
                f"got {timestep_checkpointing_interval}."
            )

        checkpoint_supports_use_reentrant = (
            "use_reentrant"
            in inspect.signature(torch.utils.checkpoint.checkpoint).parameters
        )
        checkpoint_kwargs = {}
        if checkpoint_supports_use_reentrant:
            checkpoint_kwargs["use_reentrant"] = False

        def denoise_step(
            hidden_states: torch.FloatTensor,
            step_index: int,
        ) -> torch.FloatTensor:
            t = timesteps[step_index]
            if self.interrupt:
                return hidden_states

            timestep = t.expand(hidden_states.shape[0]).to(hidden_states.dtype)
            noise_pred = self.transformer(
                hidden_states=hidden_states,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self._attention_kwargs,
                return_dict=False,
            )[0]
            if hasattr(self.scheduler, "_step_index"):
                self.scheduler._step_index = None
            output_latents = self.scheduler.step(
                noise_pred,
                t,
                hidden_states,
                return_dict=False,
            )[0]
            if (
                output_latents.dtype != hidden_states.dtype
                and torch.backends.mps.is_available()
            ):
                output_latents = output_latents.to(hidden_states.dtype)
            return output_latents

        can_checkpoint = (
            use_activation_checkpointing
            and torch.is_grad_enabled()
            and latents.requires_grad
            and callback_on_step_end is None
        )

        if can_checkpoint:
            block_size = min(timestep_checkpointing_interval, len(timesteps))
            for block_start in range(0, len(timesteps), block_size):
                block_end = min(block_start + block_size, len(timesteps))

                def _checkpointed_block(
                    hidden_states: torch.FloatTensor,
                    start=block_start,
                    end=block_end,
                ):
                    current = hidden_states
                    for step_idx in range(start, end):
                        current = denoise_step(current, step_idx)
                    return current

                latents = torch.utils.checkpoint.checkpoint(
                    _checkpointed_block,
                    latents,
                    **checkpoint_kwargs,
                )
        else:
            for i, t in enumerate(timesteps):
                latents = denoise_step(latents, i)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        if output_type == "latent":
            return latents

        latents = self._unpack_latents_with_ids(latents, latent_image_ids)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        image = self.vae.decode(latents, return_dict=False)[0]

        if output_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        else:
            image = self.image_processor.postprocess(image, output_type=output_type)

        return image
