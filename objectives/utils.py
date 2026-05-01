"""Utility functions for diversity objectives."""
from typing import Any, Dict, List, Tuple, Optional
import logging

import torch

from objectives.base_objective import BaseDiversityObjective
from objectives.lpips_diversity import LPIPSDiversityObjective
from objectives.dino_diversity import DINODiversityObjective
from objectives.dreamsim_diversity import DreamSimDiversityObjective
from objectives.tiny_l2_diversity import TinyL2DiversityObjective
from objectives.color_diversity import ColorDiversityObjective
from objectives.dpp_diversity import DPPDiversityObjective
from objectives.dpp_patch_diversity import DPPPatchDiversityObjective
from objectives.vendi_diversity import VendiDiversityObjective
from objectives.vendi_sscd_diversity import VendiSSCDDiversityObjective
from objectives.dino_backend import SharedDINOBackend
from objectives.dreamsim_backend import SharedDreamSimBackend
from objectives.sscd_backend import SharedSSCDBackend


def _create_objectives_from_spec(
    objectives_spec: Dict[str, Dict],
    device: torch.device,
    cache_dir: str,
    dino_backend: Optional[SharedDINOBackend] = None,
    dreamsim_backend: Optional[SharedDreamSimBackend] = None,
    sscd_backend: Optional[SharedSSCDBackend] = None,
    lpips_net: str = "vgg",
) -> List[BaseDiversityObjective]:
    """
    Helper to create diversity objectives from a specification dictionary.
    
    Args:
        objectives_spec: Dict mapping objective names to their config dicts
                        e.g., {'lpips': {'weighting': 1.0}, 'dino': {'weighting': 50.0}}
        device: Device for computation
        cache_dir: Cache directory
        dino_backend: Optional shared DINO backend to reuse
        dreamsim_backend: Optional shared DreamSim backend to reuse
        sscd_backend: Optional shared SSCD backend to reuse
        lpips_net: Network type for LPIPS
        
    Returns:
        List of created objectives
    """
    objectives = []
    
    for name, config in objectives_spec.items():
        weighting = config.get('weighting', 0.0)
        
        if name == 'lpips':
            objectives.append(LPIPSDiversityObjective(weighting=weighting, device=device, net=lpips_net))
        elif name == 'dino':
            objectives.append(DINODiversityObjective(weighting=weighting, device=device, dino_backend=dino_backend))
        elif name == 'dreamsim':
            objectives.append(DreamSimDiversityObjective(weighting=weighting, device=device, dreamsim_backend=dreamsim_backend, cache_dir=cache_dir))
        elif name == 'tiny_l2':
            objectives.append(TinyL2DiversityObjective(weighting=weighting, device=device))
        elif name == 'color':
            objectives.append(ColorDiversityObjective(weighting=weighting, device=device))
        elif name == 'dpp':
            backend_choice = config.get('backend', 'dino')
            objectives.append(
                DPPDiversityObjective(
                    weighting=weighting,
                    device=device,
                    dino_backend=dino_backend,
                    dreamsim_backend=dreamsim_backend,
                    backend_type=backend_choice,
                )
            )
        elif name == 'dpp_patch':
            objectives.append(
                DPPPatchDiversityObjective(
                    weighting=weighting,
                    device=device,
                    dino_backend=dino_backend,
                )
            )
        elif name == 'vendi':
            objectives.append(VendiDiversityObjective(weighting=weighting, device=device, dino_backend=dino_backend))
        elif name == 'vendi_sscd':
            objectives.append(VendiSSCDDiversityObjective(weighting=weighting, device=device, sscd_backend=sscd_backend))
    
    return objectives


def get_diversity_objectives(
    cfg: Any,
    device: torch.device,
    cache_dir: str,
) -> Tuple[List[BaseDiversityObjective], Dict[str, float]]:
    """
    Create diversity objectives from the typed Config.

    Args:
        cfg: Typed Config (see config.py)
        device: Device for computation
        cache_dir: Cache directory for model weights

    Returns:
        Tuple of (diversity objectives list, thresholds dictionary)
    """
    diversity_thresholds = {}
    objectives_spec = {}

    if cfg.diversity.lpips.enable and cfg.diversity.lpips.weight > 0:
        objectives_spec['lpips'] = {'weighting': cfg.diversity.lpips.weight}
        if cfg.diversity.lpips.threshold is not None:
            diversity_thresholds['lpips_threshold'] = cfg.diversity.lpips.threshold

    if cfg.diversity.dino.enable and cfg.diversity.dino.weight > 0:
        objectives_spec['dino'] = {'weighting': cfg.diversity.dino.weight}
        if cfg.diversity.dino.threshold is not None:
            diversity_thresholds['dino_threshold'] = cfg.diversity.dino.threshold

    if cfg.diversity.dreamsim.enable and cfg.diversity.dreamsim.weight > 0:
        objectives_spec['dreamsim'] = {'weighting': cfg.diversity.dreamsim.weight}
        if cfg.diversity.dreamsim.threshold is not None:
            diversity_thresholds['dreamsim_threshold'] = cfg.diversity.dreamsim.threshold

    if cfg.diversity.tiny_l2.enable and cfg.diversity.tiny_l2.weight > 0:
        objectives_spec['tiny_l2'] = {'weighting': cfg.diversity.tiny_l2.weight}
        if cfg.diversity.tiny_l2.threshold_multiplier is not None:
            diversity_thresholds['tiny_l2_multiplier'] = cfg.diversity.tiny_l2.threshold_multiplier

    if cfg.diversity.color.enable and cfg.diversity.color.weight > 0:
        objectives_spec['color'] = {'weighting': cfg.diversity.color.weight}
        if cfg.diversity.color.threshold_multiplier is not None:
            diversity_thresholds['color_multiplier'] = cfg.diversity.color.threshold_multiplier

    if cfg.diversity.dpp.enable and cfg.diversity.dpp.weight > 0:
        objectives_spec['dpp'] = {
            'weighting': cfg.diversity.dpp.weight,
            'backend': cfg.diversity.dpp.backend,
        }
        if cfg.diversity.dpp.threshold_multiplier is not None:
            diversity_thresholds['dpp_multiplier'] = cfg.diversity.dpp.threshold_multiplier

    if cfg.diversity.dpp_patch.enable and cfg.diversity.dpp_patch.weight > 0:
        objectives_spec['dpp_patch'] = {'weighting': cfg.diversity.dpp_patch.weight}
        if cfg.diversity.dpp_patch.threshold_multiplier is not None:
            diversity_thresholds['dpp_patch_multiplier'] = cfg.diversity.dpp_patch.threshold_multiplier

    if cfg.diversity.vendi.enable and cfg.diversity.vendi.weight > 0:
        objectives_spec['vendi'] = {'weighting': cfg.diversity.vendi.weight}
        if cfg.diversity.vendi.threshold_multiplier is not None:
            diversity_thresholds['vendi_multiplier'] = cfg.diversity.vendi.threshold_multiplier

    # Create backends for enabled objectives
    dino_backend = None
    dpp_backend_choice = objectives_spec.get('dpp', {}).get('backend', 'dino')
    dino_needs = set()
    if 'dino' in objectives_spec:
        dino_needs.add('dino')
    if 'vendi' in objectives_spec:
        dino_needs.add('vendi')
    if 'dpp' in objectives_spec and dpp_backend_choice == 'dino':
        dino_needs.add('dpp')
    if 'dpp_patch' in objectives_spec:
        dino_needs.add('dpp_patch')
    if dino_needs:
        logging.info(
            f"[SharedDINOBackend] Creating shared backend for optimization: {', '.join(sorted(dino_needs))}"
        )
        dino_backend = SharedDINOBackend(
            model_name=cfg.diversity.dino.model_name, device=device, cache_dir=cache_dir
        )

    dreamsim_backend = None
    needs_dreamsim = False
    if 'dreamsim' in objectives_spec:
        needs_dreamsim = True
    if 'dpp' in objectives_spec and dpp_backend_choice == 'dreamsim':
        needs_dreamsim = True
    if needs_dreamsim:
        logging.info("[SharedDreamSimBackend] Creating shared backend for optimization (dino_vitb16)")
        dreamsim_backend = SharedDreamSimBackend(
            device=device, cache_dir=cache_dir, backbone="dino_vitb16"
        )

    # Create objectives
    diversity_objectives = _create_objectives_from_spec(
        objectives_spec, device, cache_dir, dino_backend, dreamsim_backend, None, cfg.diversity.lpips.net
    )

    # Enable dynamic thresholds if any multiplier is set
    if any(key.endswith('_multiplier') for key in diversity_thresholds.keys()):
        diversity_thresholds['use_dynamic_thresholds'] = True
    diversity_thresholds['dynamic_threshold_multiplier'] = cfg.diversity.dynamic_threshold_multiplier

    return diversity_objectives, diversity_thresholds


def get_all_diversity_objectives(
    device: torch.device,
    cache_dir: str,
    dino_model_name: str = "facebook/dinov2-base",
    lpips_net: str = "vgg",
) -> List[BaseDiversityObjective]:
    """
    Create instances of ALL available diversity objectives for comprehensive metric logging.
    These are used for logging purposes only (computed without gradients).
    
    Args:
        device: Device for computation
        cache_dir: Cache directory for model weights
        dino_model_name: Model name for DINO-based objectives
        lpips_net: Network type for LPIPS
        
    Returns:
        List of all diversity objective instances (with weighting=0 since not used for optimization)
    """
    # Create shared backends
    logging.info("[SharedDINOBackend] Creating shared backend for comprehensive logging (DINO, DPP, Vendi)")
    dino_backend = SharedDINOBackend(model_name=dino_model_name, device=device, cache_dir=cache_dir)
    
    logging.info("[SharedDreamSimBackend] Creating shared backend for comprehensive logging (DreamSim with dino_vitb16)")
    dreamsim_backend = SharedDreamSimBackend(device=device, cache_dir=cache_dir, backbone="dino_vitb16")
    
    logging.info("[SharedSSCDBackend] Creating shared backend for comprehensive logging (SSCD)")
    sscd_backend = SharedSSCDBackend(device=device, cache_dir=cache_dir, model_name="sscd_disc_mixup")
    
    # Specification for all objectives (weighting=0 since only for logging)
    all_spec = {
        'lpips': {'weighting': 0.0},
        'dino': {'weighting': 0.0},
        'dreamsim': {'weighting': 0.0},
        'tiny_l2': {'weighting': 0.0},
        'color': {'weighting': 0.0},
        'dpp': {'weighting': 0.0, 'backend': 'dino'},
        'dpp_patch': {'weighting': 0.0},
        'vendi': {'weighting': 0.0},
        'vendi_sscd': {'weighting': 0.0},
    }
    
    return _create_objectives_from_spec(
        all_spec, device, cache_dir, dino_backend, dreamsim_backend, sscd_backend, lpips_net
    )


def get_additional_logging_objectives(
    enabled_objectives: List[BaseDiversityObjective],
    device: torch.device,
    cache_dir: str,
    dino_model_name: str = "facebook/dinov2-base",
    lpips_net: str = "vgg",
) -> Tuple[List[BaseDiversityObjective], List]:
    """
    Create diversity objectives for logging that are NOT already enabled for optimization.

    This avoids loading duplicate models when some objectives are already enabled.
    The temp_backends list is exposed so a long-lived caller (e.g. sweep runner,
    notebook) can drop references and free GPU memory between runs; for a
    single-shot script it can be ignored — process exit reclaims the memory.

    Args:
        enabled_objectives: List of objectives already loaded for optimization
        device: Device for computation
        cache_dir: Cache directory for model weights
        dino_model_name: Model name for DINO-based objectives
        lpips_net: Network type for LPIPS

    Returns:
        Tuple of (additional objectives list, temp_backends list — optional cleanup handle)
    """
    # Get names of already-enabled objectives
    enabled_names = {obj.name for obj in enabled_objectives}
    
    # All possible objectives
    all_names = {'diversity_lpips', 'diversity_dino', 'diversity_dreamsim',
                 'diversity_tiny_l2', 'diversity_color', 'diversity_dpp', 'diversity_dpp_patch',
                 'diversity_vendi', 'diversity_vendi_sscd'}
    
    # Determine which ones to load
    needed_names = all_names - enabled_names
    
    if not needed_names:
        logging.info("All diversity objectives already loaded, skipping additional loading")
        return [], []
    
    temp_backends = []
    
    # Create DINO backend if any DINO-based objectives are needed
    dino_based_needed = needed_names & {'diversity_dino', 'diversity_dpp', 'diversity_dpp_patch', 'diversity_vendi'}
    dino_backend = None
    if dino_based_needed:
        logging.info(f"[SharedDINOBackend] Creating temporary backend for logging: {', '.join(dino_based_needed)}")
        dino_backend = SharedDINOBackend(model_name=dino_model_name, device=device, cache_dir=cache_dir)
        temp_backends.append(dino_backend)
    
    # Create DreamSim backend if needed
    dreamsim_backend = None
    if 'diversity_dreamsim' in needed_names:
        logging.info("[SharedDreamSimBackend] Creating temporary backend for logging (dino_vitb16)")
        dreamsim_backend = SharedDreamSimBackend(device=device, cache_dir=cache_dir, backbone="dino_vitb16")
        temp_backends.append(dreamsim_backend)
    
    # Create SSCD backend if needed
    sscd_backend = None
    if 'diversity_vendi_sscd' in needed_names:
        logging.info("[SharedSSCDBackend] Creating temporary backend for logging (sscd_disc_mixup)")
        sscd_backend = SharedSSCDBackend(device=device, cache_dir=cache_dir, model_name="sscd_disc_mixup")
        temp_backends.append(sscd_backend)
    
    # Build specification for needed objectives
    needed_spec = {}
    name_map = {
        'diversity_lpips': 'lpips',
        'diversity_dino': 'dino',
        'diversity_dreamsim': 'dreamsim',
        'diversity_tiny_l2': 'tiny_l2',
        'diversity_color': 'color',
        'diversity_dpp': 'dpp',
        'diversity_dpp_patch': 'dpp_patch',
        'diversity_vendi': 'vendi',
        'diversity_vendi_sscd': 'vendi_sscd',
    }
    
    for full_name in needed_names:
        short_name = name_map.get(full_name)
        if short_name:
            spec_entry = {'weighting': 0.0}
            if short_name == 'dpp':
                spec_entry['backend'] = 'dino'
            needed_spec[short_name] = spec_entry
    
    additional_objectives = _create_objectives_from_spec(
        needed_spec, device, cache_dir, dino_backend, dreamsim_backend, sscd_backend, lpips_net
    )
    
    logging.info(
        f"Loaded {len(additional_objectives)} additional objectives for logging "
        f"(skipped {len(enabled_names)} already enabled)"
    )
    
    return additional_objectives, temp_backends

