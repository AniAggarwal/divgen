"""DCT-subspace genome for NSGA-II: Pareto search in the low-frequency subspace.

Motivated by two results: the paper's Fig. 9 (optimization concentrates on low
frequencies) and our CMA-DCT sweep, where a scalarized search over an 8x8
low-frequency DCT coefficient block strictly dominated full-space NSGA-II at
matched quality. Here we keep the winning parameterization and restore the
two-objective Pareto machinery.

Genome = per-slot coefficient blocks (B, C, K, K) over a per-prompt fixed white
base; phenotype latents = chi_d-repair(base + IDCT(coeffs)). Mutation is
self-adaptive Gaussian on coefficients; crossover swaps whole noise slots.
Exposes the same Population surface the evaluator and nsga2 helpers expect.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from evodiv.genome import GenomeSpec, repair


def dct_basis(K: int, H: int, device) -> torch.Tensor:
    """Orthonormal type-II DCT basis rows for the K lowest frequencies on H."""
    n = torch.arange(H, device=device).float()
    k = torch.arange(K, device=device).float()
    basis = torch.cos(math.pi * (n[None, :] + 0.5) * k[:, None] / H)
    basis[0] *= 1 / math.sqrt(2)
    return basis * math.sqrt(2.0 / H)


class DCTPopulation:
    """Population of DCT-coefficient genomes with a shared per-prompt base."""

    def __init__(self, coeffs, log_sigma, spec: GenomeSpec, base, basis):
        self.coeffs = coeffs          # (P, B, C, K, K)
        self.log_sigma = log_sigma    # (P,)
        self.spec = spec
        self.base = base              # (B, C, H, W) fixed white base
        self.basis = basis            # (K, H)
        self.F: Optional[torch.Tensor] = None
        self.raw: Optional[list] = None

    @property
    def size(self) -> int:
        return self.coeffs.shape[0]

    @property
    def latents(self) -> torch.Tensor:
        field = torch.einsum("pbckl,kh,lw->pbchw", self.coeffs, self.basis, self.basis)
        return repair(self.base.unsqueeze(0) + field)

    def sigma(self):
        return self.log_sigma.exp()

    def rate(self):
        return torch.zeros(1)

    def select(self, idx) -> "DCTPopulation":
        p = DCTPopulation(self.coeffs[idx].clone(), self.log_sigma[idx].clone(),
                          self.spec, self.base, self.basis)
        if self.F is not None:
            p.F = self.F[idx].clone()
        if self.raw is not None:
            p.raw = [self.raw[i] for i in idx.tolist()]
        return p

    @staticmethod
    def concat(a: "DCTPopulation", b: "DCTPopulation") -> "DCTPopulation":
        p = DCTPopulation(torch.cat([a.coeffs, b.coeffs], 0),
                          torch.cat([a.log_sigma, b.log_sigma], 0),
                          a.spec, a.base, a.basis)
        if a.F is not None and b.F is not None:
            p.F = torch.cat([a.F, b.F], 0)
        if a.raw is not None and b.raw is not None:
            p.raw = list(a.raw) + list(b.raw)
        return p


def init_dct_population(spec: GenomeSpec, pop_size: int, K: int, device,
                        seed: int = 0, sigma_init: float = 1.0) -> DCTPopulation:
    g = torch.Generator(device=device).manual_seed(seed)
    base = repair(torch.randn(spec.set_size, spec.channels, spec.height, spec.width,
                              device=device, generator=g))
    coeffs = sigma_init * torch.randn(pop_size, spec.set_size, spec.channels, K, K,
                                      device=device, generator=g)
    log_sigma = torch.full((pop_size,), math.log(0.5), device=device)
    basis = dct_basis(K, spec.height, device)
    return DCTPopulation(coeffs, log_sigma, spec, base, basis)


def dct_variation(pop: DCTPopulation, pa, pb, generator, tau: float = 0.3,
                  p_crossover: float = 0.9):
    """Set-level slot crossover + self-adaptive Gaussian mutation on coefficients."""
    P = pa.shape[0]
    B = pop.spec.set_size
    ca, cb = pop.coeffs[pa], pop.coeffs[pb]
    take_b = (torch.rand(P, B, device=ca.device, generator=generator) < 0.5)
    gate = (torch.rand(P, 1, device=ca.device, generator=generator) < p_crossover)
    sel = (take_b & gate).view(P, B, 1, 1, 1).float()
    child = ca * (1 - sel) + cb * sel
    ls = torch.where(gate.view(P), 0.5 * (pop.log_sigma[pa] + pop.log_sigma[pb]), pop.log_sigma[pa])
    ls = ls + tau * torch.randn(P, device=ca.device, generator=generator)
    sigma = ls.exp().clamp(0.02, 4.0).view(P, 1, 1, 1, 1)
    child = child + sigma * torch.randn(child.shape, device=ca.device, generator=generator)
    return DCTPopulation(child, ls, pop.spec, pop.base, pop.basis)
