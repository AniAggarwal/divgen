"""Utility functions for different noise initialization strategies."""

import logging

import torch
import numpy as np


def _generate_2d_pink_noise(batch_size, channels, height, width, device, dtype, generator, exponent):
    """
    Generate 2D pink noise via spectral filtering: 1/(1+f)^exponent.

    Matches the paper formulation:
        z_white ~ N(0, I)
        z_hat   = FFT2D(z_white)
        z_hat_pink(u, v) = z_hat(u, v) / (1 + sqrt(u^2 + v^2))^alpha
        z_pink  = (IFFT2D(z_hat_pink) - mu) / sigma     (per-channel)
    The (1+f) term keeps DC at unit gain, so no explicit DC zeroing is needed.
    """
    pink_noise = torch.randn(batch_size, channels, height, width, device='cpu',
                             dtype=torch.float32, generator=generator)

    if exponent == 0.0:
        # White noise: still apply per-channel mean/std normalization for consistency
        flat = pink_noise.reshape(batch_size * channels, height * width)
        flat = (flat - flat.mean(dim=1, keepdim=True)) / (flat.std(dim=1, keepdim=True) + 1e-8)
        return flat.reshape(batch_size, channels, height, width).to(device=device, dtype=dtype)

    # Radial frequency grid in integer index units (matches u, v in the paper).
    freq_h = torch.fft.fftfreq(height).view(height, 1) * height
    freq_w = torch.fft.rfftfreq(width).view(1, width // 2 + 1) * width
    freq_radial = torch.sqrt(freq_h**2 + freq_w**2)

    scaling = 1.0 / (1.0 + freq_radial) ** exponent

    for b in range(batch_size):
        for c in range(channels):
            noise_fft = torch.fft.rfft2(pink_noise[b, c])
            pink_noise[b, c] = torch.fft.irfft2(noise_fft * scaling, s=(height, width))

    pink_noise = pink_noise.reshape(batch_size * channels, height * width)
    pink_noise = (pink_noise - pink_noise.mean(dim=1, keepdim=True)) / (pink_noise.std(dim=1, keepdim=True) + 1e-8)
    pink_noise = pink_noise.reshape(batch_size, channels, height, width)

    return pink_noise.to(device=device, dtype=dtype)


def _seq_len_to_2d(seq_len):
    """Convert a flattened sequence length to 2D spatial dimensions."""
    sqrt_seq = int(seq_len ** 0.5)
    if sqrt_seq * sqrt_seq == seq_len:
        return sqrt_seq, sqrt_seq
    for h in [16, 32, 64, 128]:
        if seq_len % h == 0:
            w = seq_len // h
            if w <= 256:
                return h, w
    return sqrt_seq, seq_len // sqrt_seq


def generate_pink_noise(shape, device, dtype=torch.float32, seed=None, exponent=1.0):
    """
    Generate pink noise using 1/(1+f)^exponent spectral filtering on a 2D grid.

    Per-sample seeding (seed + i) matches the white-noise convention so the only
    difference between white and pink for the same seed is the spectral filter.

    Args:
        shape: (B, seq_len, C) for sequence-shaped latents (3D path), or
               (B, C, H, W) for image-shaped latents (4D path).
        device: Device to create tensor on.
        dtype: Data type of the tensor.
        seed: Random seed for reproducibility (optional).
        exponent: Amplitude spectrum exponent (0=white, 1=pink). alpha in [0, 1].

    Returns:
        Tensor of pink noise with the specified shape.
    """
    if len(shape) == 3:
        # Sequence-shaped latents: route through a 2D grid so spatial correlations
        # survive when the consumer interprets the sequence as packed image tokens.
        batch_size, seq_len, channels = shape
        height, width = _seq_len_to_2d(seq_len)

        samples = []
        for i in range(batch_size):
            gen = torch.Generator(device='cpu')
            if seed is not None:
                gen.manual_seed(seed + i)
            pink_2d = _generate_2d_pink_noise(1, channels, height, width,
                                              device, dtype, gen, exponent)
            pink_seq = pink_2d.permute(0, 2, 3, 1).reshape(1, seq_len, channels).contiguous()
            samples.append(pink_seq)
        return torch.cat(samples, dim=0)

    elif len(shape) == 4:
        batch_size, channels, height, width = shape

        samples = []
        for i in range(batch_size):
            gen = torch.Generator(device='cpu')
            if seed is not None:
                gen.manual_seed(seed + i)
            samples.append(_generate_2d_pink_noise(1, channels, height, width,
                                                    device, dtype, gen, exponent))
        return torch.cat(samples, dim=0)

    else:
        raise ValueError(f"Unsupported shape: {shape}. Expected 3D or 4D tensor.")


def generate_latents(shape, device, dtype, noise_type='white', seed=None, noise_exponent=1.0,
                     flux_schnell_pack=False):
    """
    Generate initial latents with specified noise type.

    Args:
        shape: Shape of the latent tensor. For Flux-schnell with `flux_schnell_pack=True`,
            pass the 4D pre-pack shape (B, 16, 64, 64) at 512x512; the result is packed
            to (B, 1024, 64). For Flux-klein, pass the 4D pre-pack shape and leave
            `flux_schnell_pack=False` (the klein pipeline packs internally), so the 2D
            spatial pink correlations are preserved end-to-end.
        device: Device to create tensor on.
        dtype: Data type of the tensor.
        noise_type: 'white' or 'pink'.
        seed: Random seed for reproducibility (optional).
        noise_exponent: Amplitude spectrum exponent for pink noise (alpha in [0, 1]).
        flux_schnell_pack: If True, generate noise in the given 4D shape, then apply
            Flux's 2x2 spatial->channel packing so spatial pink correlations survive
            into the packed sequence consumed by Flux-schnell.

    Returns:
        Tensor of latents with specified noise type.
    """
    if noise_type == 'white':
        if seed is not None:
            # Per-sample CPU generator (seed + i) so white and pink share the seeding
            # convention.
            batch_size = shape[0]
            single_shape = (1,) + tuple(shape[1:])
            samples = []
            for i in range(batch_size):
                gen = torch.Generator(device='cpu').manual_seed(seed + i)
                samples.append(torch.randn(single_shape, device='cpu', dtype=dtype, generator=gen))
            latents = torch.cat(samples, dim=0).to(device)
        else:
            latents = torch.randn(shape, device=device, dtype=dtype)
    elif noise_type == 'pink':
        latents = generate_pink_noise(shape, device, dtype, seed=seed, exponent=noise_exponent)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose 'white' or 'pink'.")

    if flux_schnell_pack:
        if latents.dim() != 4:
            raise ValueError(
                f"flux_schnell_pack=True requires a 4D shape (B, C, H, W); got {tuple(latents.shape)}"
            )
        b, c, h, w = latents.shape
        # 2x2 spatial -> channel packing, matching diffusers FluxPipeline._pack_latents.
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)

    if noise_type == 'pink':
        l32 = latents.detach().to(torch.float32)
        m, s = l32.mean().item(), l32.std().item()
        if not (np.isfinite(m) and np.isfinite(s)) or abs(m) > 0.05 or not (0.9 < s < 1.1):
            logging.warning(
                f"[generate_latents] pink stats off: mean={m:.4f} std={s:.4f} "
                f"(expected ~0 / ~1; exponent={noise_exponent})"
            )

    return latents
