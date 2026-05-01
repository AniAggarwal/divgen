"""Typed configuration for divgen.

The `Config` dataclass tree is the source of truth. YAML files load directly
into it via `Config.from_yaml`. The CLI is generated from the dataclass tree
(nested flag names like `--rewards.hps.enable`) and merged onto the YAML
defaults.
"""
import argparse
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    cache_dir: str = "./cache"
    save_dir: str = "./outputs"


@dataclass
class ModelConfig:
    name: str = "sdxl-turbo"
    dtype: str = "float16"
    device_id: Optional[int] = None
    cpu_offloading: bool = False


@dataclass
class OptimizationConfig:
    lr: float = 5.0
    n_iters: int = 50
    n_inference_steps: int = 1
    num_samples: int = 1
    sequential_diverse_samples: int = 0
    sequential_diversity_objective: str = "diversity_dpp"
    optim: str = "sgd"
    grad_clip: float = 0.1
    seed: int = 0
    noise_type: str = "white"
    noise_exponent: float = 0.2


@dataclass
class RegularizationConfig:
    enable: bool = False
    weight: float = 0.01


@dataclass
class HPSRewardConfig:
    enable: bool = False
    weighting: float = 5.0
    warmup_iters: int = 0
    enable_stopping: bool = False
    enable_revert: bool = False
    stopping_threshold: float = 0.3
    absolute_threshold: Optional[float] = None


@dataclass
class ImageRewardRewardConfig:
    enable: bool = False
    weighting: float = 1.0


@dataclass
class CLIPRewardConfig:
    enable: bool = False
    weighting: float = 0.01


@dataclass
class CLIPB32RewardConfig:
    enable: bool = False
    weighting: float = 0.01


@dataclass
class PickScoreRewardConfig:
    enable: bool = False
    weighting: float = 0.05


@dataclass
class RewardsConfig:
    hps: HPSRewardConfig = field(default_factory=HPSRewardConfig)
    imagereward: ImageRewardRewardConfig = field(default_factory=ImageRewardRewardConfig)
    clip: CLIPRewardConfig = field(default_factory=CLIPRewardConfig)
    clip_b32: CLIPB32RewardConfig = field(default_factory=CLIPB32RewardConfig)
    pickscore: PickScoreRewardConfig = field(default_factory=PickScoreRewardConfig)


@dataclass
class LPIPSDivConfig:
    enable: bool = False
    weight: float = 0.0
    net: str = "vgg"
    threshold: Optional[float] = None


@dataclass
class DINODivConfig:
    enable: bool = False
    weight: float = 0.0
    model_name: str = "facebook/dinov2-base"
    threshold: Optional[float] = None


@dataclass
class DreamSimDivConfig:
    enable: bool = False
    weight: float = 0.0
    threshold: Optional[float] = None


@dataclass
class TinyL2DivConfig:
    enable: bool = False
    weight: float = 0.0
    threshold_multiplier: Optional[float] = None


@dataclass
class ColorDivConfig:
    enable: bool = False
    weight: float = 0.0
    threshold_multiplier: Optional[float] = None


@dataclass
class DPPDivConfig:
    enable: bool = False
    weight: float = 0.0
    backend: str = "dino"
    threshold_multiplier: Optional[float] = None


@dataclass
class DPPPatchDivConfig:
    enable: bool = False
    weight: float = 0.0
    threshold_multiplier: Optional[float] = None


@dataclass
class VendiDivConfig:
    enable: bool = False
    weight: float = 0.0
    threshold_multiplier: Optional[float] = None


@dataclass
class VendiSSCDDivConfig:
    """SSCD model name override (the vendi_sscd objective is enabled via vendi)."""
    model_name: str = "sscd_disc_mixup"


@dataclass
class DiversityConfig:
    lpips: LPIPSDivConfig = field(default_factory=LPIPSDivConfig)
    dino: DINODivConfig = field(default_factory=DINODivConfig)
    dreamsim: DreamSimDivConfig = field(default_factory=DreamSimDivConfig)
    tiny_l2: TinyL2DivConfig = field(default_factory=TinyL2DivConfig)
    color: ColorDivConfig = field(default_factory=ColorDivConfig)
    dpp: DPPDivConfig = field(default_factory=DPPDivConfig)
    dpp_patch: DPPPatchDivConfig = field(default_factory=DPPPatchDivConfig)
    vendi: VendiDivConfig = field(default_factory=VendiDivConfig)
    vendi_sscd: VendiSSCDDivConfig = field(default_factory=VendiSSCDDivConfig)
    dynamic_threshold_multiplier: float = 4.0


@dataclass
class StoppingConfig:
    enable_dpp_stopping: bool = False
    dpp_stopping_threshold: float = 0.03
    dpp_stopping_multiplier: Optional[float] = None
    dpp_stopping_absolute: Optional[float] = None


@dataclass
class TaskConfig:
    type: str = "single"
    prompt: str = "A red dog and a green cat"
    prompt_start_index: Optional[int] = None
    prompt_end_index: Optional[int] = None
    t2i_subset: str = "all"


@dataclass
class MultiStepConfig:
    enable: bool = False


@dataclass
class LoggingConfig:
    save_iteration_history: bool = False


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    stopping: StoppingConfig = field(default_factory=StoppingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    multi_step: MultiStepConfig = field(default_factory=MultiStepConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        if self.rewards.hps.enable_revert and not self.rewards.hps.enable_stopping:
            raise ValueError("rewards.hps.enable_revert requires rewards.hps.enable_stopping=True")
        if self.model.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Invalid model.dtype: {self.model.dtype!r}")
        if self.optimization.optim not in ("sgd", "adam", "lbfgs"):
            raise ValueError(f"Invalid optimization.optim: {self.optimization.optim!r}")
        if self.optimization.noise_type not in ("white", "pink"):
            raise ValueError(f"Invalid noise_type: {self.optimization.noise_type!r}")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML required: pip install pyyaml") from e
        with open(path) as f:
            return _build(cls, yaml.safe_load(f) or {}, cls.__name__)

    @classmethod
    def from_args(cls, argv: Optional[List[str]] = None) -> "Config":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--config", type=str, default=None)
        pre_args, _ = pre.parse_known_args(argv)
        if pre_args.config is not None:
            if not os.path.exists(pre_args.config):
                raise FileNotFoundError(f"Config file not found: {pre_args.config}")
            base = cls.from_yaml(pre_args.config)
            # CLI startup message: runs before main() configures logging, so use print.
            print(f"Loaded configuration from: {pre_args.config}")
        else:
            base = cls()
        ns = _build_parser(cls).parse_args(argv)
        return _apply_overrides(base, _ns_to_dict(ns))


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _unwrap(tp):
    if get_origin(tp) is Union:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _build(cls, data: Any, path: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {path}, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys at {path}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value, inner = data[f.name], _unwrap(hints[f.name])
        if is_dataclass(inner):
            kwargs[f.name] = _build(inner, value, f"{path}.{f.name}")
        else:
            if value is not None:
                if inner is float and isinstance(value, int) and not isinstance(value, bool):
                    value = float(value)
                elif inner is int and isinstance(value, bool):
                    raise TypeError(f"{path}.{f.name}: expected int, got bool")
                elif not isinstance(value, inner):
                    raise TypeError(f"{path}.{f.name}: expected {inner.__name__}, got {type(value).__name__}")
            kwargs[f.name] = value
    return cls(**kwargs)


def _build_parser(cls) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="divgen configuration")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (CLI args override YAML)")
    _add_args(p, cls, "")
    return p


def _add_args(parser, cls, prefix: str):
    hints = get_type_hints(cls)
    for f in fields(cls):
        inner = _unwrap(hints[f.name])
        dotted = f"{prefix}{f.name}"
        if is_dataclass(inner):
            _add_args(parser, inner, f"{dotted}.")
            continue
        if inner is bool:
            grp = parser.add_mutually_exclusive_group()
            grp.add_argument(f"--{dotted}", dest=dotted, action="store_true", default=argparse.SUPPRESS)
            grp.add_argument(f"--no-{dotted}", dest=dotted, action="store_false", default=argparse.SUPPRESS)
        else:
            kw = {"default": argparse.SUPPRESS}
            if inner in (int, float, str):
                kw["type"] = inner
            parser.add_argument(f"--{dotted}", dest=dotted, **kw)


def _ns_to_dict(ns: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(ns).items():
        if k == "config":
            continue
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def _apply_overrides(cfg, overrides: Dict[str, Any]):
    if not overrides or not is_dataclass(cfg):
        return overrides if overrides else cfg
    changes = {}
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown override key: {k}")
        cur = getattr(cfg, k)
        changes[k] = _apply_overrides(cur, v) if is_dataclass(cur) and isinstance(v, dict) else v
    return replace(cfg, **changes)
