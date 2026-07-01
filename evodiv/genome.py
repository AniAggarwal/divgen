"""Noise-set genomes and evolutionary operators for EvoDiv.

The central object of EvoDiv is a *set genome*: a batch of ``B`` initial-noise
latents for a single prompt. Because the diversity metrics reported by the
"It's Never Too Late" paper (Tab. 1/2) are *set-level* quantities (averaged
pairwise DINOv2 / DreamSim / LPIPS, DPP, Vendi over a set of 4 images), the
natural unit of selection is a whole set of noises rather than a single noise.

A population of ``P`` genomes is stored as a handful of stacked tensors so that
all evolutionary operators are vectorised across the population:

    latents     (P, B, C, H, W)  float32  -- master noise (cast to model dtype at render time)
    log_sigma   (P,)             float32  -- self-adaptive mutation step size (log domain)
    logit_rate  (P,)             float32  -- self-adaptive per-element mutation rate (logit domain)

Design choices, each tied to a finding in the source papers:

* **Spectral, low-frequency-biased mutation.** The paper's Fig. 9 shows noise
  optimisation acts predominantly on the *low third* of the power spectrum. We
  therefore colour every mutation with a ``1/(1+f)^beta`` filter so proposals
  move low frequencies more than high ones -- the evolutionary analogue of
  their gradient dynamics, and the same filter used for their pink-noise init.

* **chi_d-norm repair.** The paper keeps noises in a high-density region of the
  Gaussian prior via a soft regulariser ``K(eps)`` that pins ``||eps||`` to the
  chi_d radius. We enforce the same constraint *exactly* by standardising each
  latent to mean 0 / std 1 after every operator (a repair operator), so
  ``||eps|| ~ sqrt(d)`` by construction -- the mode of the chi_d law.

* **Self-adaptive mutation rate + step (requested idea 3).** ECAD's Table 9
  shows 1% mutation stalls in bad optima and 15% converges too slowly, with the
  sweet spot unknown. Rather than fix it, each genome *carries its own* rate and
  step and they evolve alongside the noise (log-normal self-adaptation, as in
  Evolution Strategies), so the population discovers its own schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from training.noise_utils import generate_latents


@dataclass
class GenomeSpec:
    """Static description of a noise-set genome and its operator hyperparameters."""

    channels: int
    height: int
    width: int
    set_size: int = 4                     # B: images optimised jointly per prompt
    beta: float = 1.0                     # spectral colouring exponent for mutations (low-freq bias)
    sigma_init: float = 0.30              # initial mutation step size
    sigma_bounds: Tuple[float, float] = (0.02, 1.0)
    rate_init: float = 0.10               # initial per-element mutation probability
    rate_bounds: Tuple[float, float] = (0.01, 0.50)
    tau_sigma: float = 0.30               # log-normal self-adaptation rate for sigma
    tau_rate: float = 0.30                # logit self-adaptation rate for the mutation rate
    self_adaptive: bool = True            # if False, sigma/rate stay fixed at their init values

    @property
    def dim(self) -> int:
        return self.channels * self.height * self.width

    @property
    def latent_shape(self) -> Tuple[int, int, int, int]:
        return (self.set_size, self.channels, self.height, self.width)


# --------------------------------------------------------------------------- #
# Core tensor helpers                                                          #
# --------------------------------------------------------------------------- #


def _radial_frequency(height: int, width: int, device) -> torch.Tensor:
    """Radial frequency grid f = sqrt(u^2 + v^2) for a 2D rFFT, in index units.

    Matches the paper's pink-noise construction (training/noise_utils.py).
    """
    freq_h = torch.fft.fftfreq(height, device=device).view(height, 1) * height
    freq_w = torch.fft.rfftfreq(width, device=device).view(1, width // 2 + 1) * width
    return torch.sqrt(freq_h ** 2 + freq_w ** 2)


def spectral_color(white: torch.Tensor, beta: float) -> torch.Tensor:
    """Colour white noise with a 1/(1+f)^beta spectral filter (per-sample).

    ``white`` has shape (..., C, H, W). Returns a tensor of the same shape,
    standardised per-sample to unit variance so ``sigma`` remains an
    interpretable step size regardless of ``beta``.
    """
    *lead, c, h, w = white.shape
    flat = white.reshape(-1, c, h, w)
    if beta != 0.0:
        scale = 1.0 / (1.0 + _radial_frequency(h, w, white.device)) ** beta
        fftd = torch.fft.rfft2(flat.float())
        flat = torch.fft.irfft2(fftd * scale, s=(h, w)).to(white.dtype)
    # standardise each (C,H,W) sample to unit std so step size is comparable across beta
    flat = flat.reshape(flat.shape[0], -1)
    flat = flat / (flat.std(dim=1, keepdim=True) + 1e-8)
    return flat.reshape(*lead, c, h, w)


def repair(latents: torch.Tensor) -> torch.Tensor:
    """chi_d-norm repair: standardise each latent to mean 0 / std 1.

    Keeps every noise on the high-density shell of the Gaussian prior
    (||eps|| ~ sqrt(d)), the exact target of the paper's K(eps) regulariser,
    but enforced as a hard repair rather than a soft penalty.
    """
    lead = latents.shape[:-3]
    flat = latents.reshape(*lead, -1)
    flat = (flat - flat.mean(dim=-1, keepdim=True)) / (flat.std(dim=-1, keepdim=True) + 1e-8)
    return flat.reshape(latents.shape)


def sigma_from(log_sigma: torch.Tensor, spec: GenomeSpec) -> torch.Tensor:
    lo, hi = spec.sigma_bounds
    return log_sigma.exp().clamp(lo, hi)


def rate_from(logit_rate: torch.Tensor, spec: GenomeSpec) -> torch.Tensor:
    lo, hi = spec.rate_bounds
    return (lo + (hi - lo) * torch.sigmoid(logit_rate))


# --------------------------------------------------------------------------- #
# Population construction                                                      #
# --------------------------------------------------------------------------- #


class Population:
    """A vectorised population of noise-set genomes.

    Attributes are plain tensors so every operator is a batched tensor op.
    ``F`` holds the (P, 2) objective matrix (minimisation convention:
    ``[-quality, -diversity]``) once the population has been evaluated.
    """

    def __init__(self, latents, log_sigma, logit_rate, spec: GenomeSpec):
        self.latents = latents          # (P, B, C, H, W) float32
        self.log_sigma = log_sigma      # (P,)
        self.logit_rate = logit_rate    # (P,)
        self.spec = spec
        self.F: Optional[torch.Tensor] = None   # (P, 2) objectives, minimisation
        self.raw: Optional[dict] = None         # per-genome raw metric dict list

    @property
    def size(self) -> int:
        return self.latents.shape[0]

    def sigma(self) -> torch.Tensor:
        return sigma_from(self.log_sigma, self.spec)

    def rate(self) -> torch.Tensor:
        return rate_from(self.logit_rate, self.spec)

    def select(self, idx: torch.Tensor) -> "Population":
        p = Population(
            self.latents[idx].clone(),
            self.log_sigma[idx].clone(),
            self.logit_rate[idx].clone(),
            self.spec,
        )
        if self.F is not None:
            p.F = self.F[idx].clone()
        if self.raw is not None:
            p.raw = [self.raw[i] for i in idx.tolist()]
        return p

    @staticmethod
    def concat(a: "Population", b: "Population") -> "Population":
        p = Population(
            torch.cat([a.latents, b.latents], 0),
            torch.cat([a.log_sigma, b.log_sigma], 0),
            torch.cat([a.logit_rate, b.logit_rate], 0),
            a.spec,
        )
        if a.F is not None and b.F is not None:
            p.F = torch.cat([a.F, b.F], 0)
        if a.raw is not None and b.raw is not None:
            p.raw = list(a.raw) + list(b.raw)
        return p


def init_population(
    spec: GenomeSpec,
    pop_size: int,
    device,
    noise_type: str = "white",
    noise_exponent: float = 0.2,
    seed: int = 0,
) -> Population:
    """Sample an initial population of ``pop_size`` noise-set genomes.

    Each genome's ``B`` latents are drawn i.i.d. from the same white/pink
    initialisation used by the base method (``training.noise_utils``), so
    generation 0 *is* the paper's i.i.d. baseline -- evolution then improves on
    it. Different genomes use disjoint seed offsets to avoid duplicates.
    """
    lat = torch.empty((pop_size,) + spec.latent_shape, dtype=torch.float32, device=device)
    for i in range(pop_size):
        g = generate_latents(
            spec.latent_shape, device, torch.float32,
            noise_type=noise_type, seed=seed + i * spec.set_size,
            noise_exponent=noise_exponent,
        )
        lat[i] = repair(g)
    import math
    log_sigma = torch.full((pop_size,), math.log(spec.sigma_init), device=device)
    lo, hi = spec.rate_bounds
    r0 = min(max(spec.rate_init, lo + 1e-4), hi - 1e-4)
    logit0 = math.log((r0 - lo) / (hi - r0))
    logit_rate = torch.full((pop_size,), logit0, device=device)
    return Population(lat, log_sigma, logit_rate, spec)


# --------------------------------------------------------------------------- #
# Evolutionary operators (vectorised over offspring)                          #
# --------------------------------------------------------------------------- #


def mutate(
    latents: torch.Tensor,
    log_sigma: torch.Tensor,
    logit_rate: torch.Tensor,
    spec: GenomeSpec,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Self-adaptive, spectrally-coloured mutation over a batch of genomes.

    Steps (all vectorised over the leading population dimension):
      1. Self-adapt the strategy params first (so offspring are evaluated with
         the step that produced them -- standard ES ordering):
             log_sigma'  = log_sigma  + tau_sigma * N(0,1)
             logit_rate' = logit_rate + tau_rate  * N(0,1)
      2. Draw a spectrally-coloured perturbation (low-freq boosted) of unit std.
      3. Apply it under a per-element Bernoulli(rate) mask, scaled by sigma
         (bit-flip-style sparse mutation, but in continuous noise space).
      4. chi_d-norm repair.
    """
    device = latents.device
    P = latents.shape[0]

    if spec.self_adaptive:
        n_sigma = torch.randn(P, device=device, generator=generator)
        n_rate = torch.randn(P, device=device, generator=generator)
        log_sigma = log_sigma + spec.tau_sigma * n_sigma
        logit_rate = logit_rate + spec.tau_rate * n_rate

    sigma = sigma_from(log_sigma, spec).view(P, 1, 1, 1, 1)
    rate = rate_from(logit_rate, spec).view(P, 1, 1, 1, 1)

    white = torch.randn(latents.shape, device=device, generator=generator)
    delta = spectral_color(white, spec.beta)
    mask = (torch.rand(latents.shape, device=device, generator=generator) < rate).float()

    child = latents + mask * sigma * delta
    child = repair(child)
    return child, log_sigma, logit_rate


def crossover(
    latents: torch.Tensor,
    log_sigma: torch.Tensor,
    logit_rate: torch.Tensor,
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    spec: GenomeSpec,
    generator: torch.Generator,
    p_crossover: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set-level uniform crossover between parent pairs (idx_a, idx_b).

    Each of the ``B`` noise slots of the child is inherited (whole) from parent
    A or parent B independently -- recombination in the space of *sets*, which
    respects the fact that individual noises are the atoms of diversity.
    Strategy parameters recombine by intermediate (average) recombination.
    A per-pair Bernoulli(p_crossover) gate decides whether crossover happens at
    all; otherwise the child is a clone of parent A.
    """
    device = latents.device
    n = idx_a.shape[0]
    B = spec.set_size

    la, lb = latents[idx_a], latents[idx_b]
    # per (pair, slot) choice of parent
    take_b = (torch.rand(n, B, device=device, generator=generator) < 0.5)
    gate = (torch.rand(n, 1, device=device, generator=generator) < p_crossover)
    take_b = take_b & gate
    sel = take_b.view(n, B, 1, 1, 1).float()
    child = la * (1.0 - sel) + lb * sel
    child = repair(child)

    # intermediate recombination of strategy params, gated the same way
    g = gate.view(n).float()
    ls = torch.where(g.bool(), 0.5 * (log_sigma[idx_a] + log_sigma[idx_b]), log_sigma[idx_a])
    lr = torch.where(g.bool(), 0.5 * (logit_rate[idx_a] + logit_rate[idx_b]), logit_rate[idx_a])
    return child, ls, lr
