"""Configuration for EvoDiv.

Reuses divgen's typed-dataclass config tree (model / rewards / diversity /
paths) verbatim and adds an ``evolution`` section. The CLI + YAML plumbing is
inherited from ``config.py`` (nested flags like ``--evolution.pop_size``).
"""

from dataclasses import dataclass, field
from typing import Optional

from config import (  # divgen's config module
    Config,
    _apply_overrides,
    _build_parser,
    _ns_to_dict,
)
import argparse
import os


@dataclass
class EvolutionConfig:
    # population / budget
    pop_size: int = 32
    n_generations: int = 40
    set_size: int = 4                 # B images optimised jointly per prompt
    seed: int = 0
    render_chunk: int = 128           # max images per forward pass (memory cap)

    # genome + operators
    genome: str = "noise"             # "noise" (spectral set genome) or "gp"
    beta: float = 1.0                 # spectral colouring exponent (low-freq bias) for mutations
    sigma_init: float = 0.30
    sigma_min: float = 0.02
    sigma_max: float = 1.0
    rate_init: float = 0.10
    rate_min: float = 0.01
    rate_max: float = 0.50
    self_adaptive: bool = True        # each genome evolves its own sigma + mutation rate
    tau_sigma: float = 0.30
    tau_rate: float = 0.30
    p_crossover: float = 0.9
    repair_ops: bool = True           # chi_d-norm repair after operators (ablatable)

    # island model (requested idea 1); prompt sets are comma-separated files or names
    islands: int = 1
    migration_interval: int = 5
    migration_size: int = 2
    island_prompt_files: str = ""     # comma-separated prompt files, one per island
    island_eval_prompts: int = 3      # max prompts sampled per island for fitness

    # selection of the reported genome
    select_metric: str = "diversity_dino"   # raw metric maximised when picking the winner
    quality_floor_frac: float = 0.98        # winner must keep >= frac * gen0 mean quality

    # early stopping (large sweeps); 0 / None disables
    early_stop_patience: int = 0            # gens without best-div improvement before stopping
    early_stop_min_delta: float = 0.002
    target_diversity: Optional[float] = None
    target_quality_frac: Optional[float] = None  # target quality = frac * gen0 MEAN quality

    # resume: skip prompts whose history.json already exists
    skip_existing: bool = True

    # B200 throughput options
    use_surrogate: bool = False       # TAESD decode during search gens (exact at gen0/anchors/final)
    exact_every: int = 10             # re-anchor survivors with the exact decoder every k gens
    compile_model: bool = False       # torch.compile the UNet

    # logging
    log_all_every: int = 0            # also compute held-out metrics every k gens (0 = only last)


@dataclass
class EvoConfig(Config):
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    @classmethod
    def from_args(cls, argv=None) -> "EvoConfig":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", type=str, default=None)
        pre_args, _ = pre.parse_known_args(argv)
        if pre_args.config is not None:
            if not os.path.exists(pre_args.config):
                raise FileNotFoundError(f"Config file not found: {pre_args.config}")
            base = cls.from_yaml(pre_args.config)
            print(f"Loaded configuration from: {pre_args.config}")
        else:
            base = cls()
        ns = _build_parser(cls).parse_args(argv)
        return _apply_overrides(base, _ns_to_dict(ns))
