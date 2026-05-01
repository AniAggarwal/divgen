from typing import List, Optional, Union

import torch
from diffusers import PixArtAlphaPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    retrieve_timesteps,
)

from models.utils import freeze_params


class RewardPixartPipeline(PixArtAlphaPipeline):
    def __init__(
        self, tokenizer, text_encoder, transformer, scheduler, vae
    ):
        super().__init__(
            tokenizer,
            text_encoder,
            vae,
            transformer,
            scheduler,
        )
        self.text_encoder.gradient_checkpointing_enable()
        self.vae.enable_gradient_checkpointing()
        self.text_encoder.eval()
        self.vae.eval()
        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())

    def apply(
        self,
        latents: torch.Tensor = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = [400],
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        callback_steps: int = 1,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs,
    ):
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(
                batch_size * num_images_per_prompt, 1
            )
            aspect_ratio = torch.tensor([float(height / width)]).repeat(
                batch_size * num_images_per_prompt, 1
            )
            resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
            aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

            if do_classifier_free_guidance:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor(
                    [current_timestep], dtype=dtype, device=latent_model_input.device
                )
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            if num_inference_steps == 1:
                # For DMD one step sampling: https://arxiv.org/abs/2311.18828
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).pred_original_sample

        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        if use_resolution_binning:
            image = self.image_processor.resize_and_crop_tensor(
                image, orig_width, orig_height
            )

        image = (image / 2 + 0.5).clamp(0, 1)

        # Offload all models
        self.maybe_free_model_hooks()
        return image

