# Night-queue experiments (exploratory — for verification)

Extra genetic-method runs launched after the planned E1–E11 to keep the
paid GPU busy. Forward-only, SDXL-Turbo, surrogate+compile, resume-safe,
never bred on ImageReward. **Not folded into the paper's headline numbers**
— review and tell me which to promote (each has saved image sets + history).

Data: `runs/paperprep/night/<name>/`, mirrored to Tigris
`paperprep-night-<name>.tar.gz`.

## Spectral-bias (beta) curve, 64 prompts
*beta {0,0.5,1(=main),2}*

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NA_beta0.0 | 64 | 0.712 | 0.352 | 0.677 | 0.340 | 2.370 | 0.812 |
| NA_beta0.5 | 64 | 0.726 | 0.373 | 0.685 | 0.341 | 2.450 | 0.824 |
| NA_beta2.0 | 64 | 0.829 | 0.618 | 0.804 | 0.334 | 3.234 | 0.922 |

## Set-size / diversity ceiling, 64 prompts
*B images per set*

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NB_B16 | 64 | 0.747 | 0.473 | 0.775 | 0.333 | 6.193 | 0.683 |
| NB_B8 | 64 | 0.755 | 0.453 | 0.746 | 0.340 | 4.113 | 0.762 |

## Which diversity objective breeds best, 64 prompts
*bred objective*

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NC_dppobj | 64 | 0.679 | 0.429 | 0.679 | 0.340 | 3.201 | 0.917 |
| NC_vendiobj | 64 | 0.694 | 0.440 | 0.688 | 0.340 | 3.274 | 0.925 |

## Pink vs white initialization, 128 prompts

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| ND_pink | 128 | 0.794 | 0.453 | 0.782 | 0.365 | 2.853 | 0.877 |

## CMA-DCT dimensionality sweep, 40 prompts
*K = DCT block edge; DIM=4*4*K^2*

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NE_cmaK12 | 40 | 0.689 | 0.337 | 0.865 | 0.320 | 2.774 | 0.870 |
| NE_cmaK4 | 40 | 0.810 | 0.537 | 0.825 | 0.358 | 3.022 | 0.898 |
| NE_cmaK6 | 40 | 0.800 | 0.500 | 0.841 | 0.354 | 2.976 | 0.893 |

## GP program-synthesis genome, 32 prompts

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NF_gp | 32 | 0.866 | 0.717 | 0.860 | 0.240 | 3.642 | 0.967 |

## Crossover / repair ablations at scale, 64 prompts

| run | n | DINO | DreamSim | LPIPS | CLIP | Vendi | DPP |
|---|--:|--:|--:|--:|--:|--:|--:|
| NG_norepair | 64 | 0.776 | 0.474 | 0.735 | 0.335 | 2.861 | 0.875 |
| NG_noxover | 64 | 0.798 | 0.539 | 0.811 | 0.332 | 2.978 | 0.891 |

