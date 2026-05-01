"""Early stopping utilities for diversity-based training."""
import logging
import math
from typing import Callable, Dict, Optional, Tuple


def calculate_dynamic_thresholds(
    rewards: Dict[str, float],
    diversity_thresholds: Dict[str, float],
    iteration: int,
) -> Dict[str, float]:
    """
    Calculate dynamic thresholds based on initial diversity values.
    
    Args:
        rewards: Dictionary of current rewards/diversity scores
        diversity_thresholds: Dictionary containing multiplier settings
        iteration: Current iteration number (0 = first iteration)
    
    Returns:
        Dictionary mapping metric names to calculated threshold values
    """
    if iteration != 0:
        return {}
    
    use_dynamic = diversity_thresholds.get('use_dynamic_thresholds', False)
    if not use_dynamic:
        return {}
    
    default_multiplier = diversity_thresholds.get('dynamic_threshold_multiplier', 4.0)
    
    # Metrics that use dynamic thresholds with their default multipliers
    dynamic_metrics = {
        "diversity_tiny_l2": ("tiny_l2_multiplier", 1.5),  # Default 1.5 for Tiny L2
        "diversity_color": ("color_multiplier", 1.5),  # Default 1.5 for Color
        "diversity_dpp": ("dpp_multiplier", default_multiplier),
        "diversity_dpp_patch": ("dpp_patch_multiplier", default_multiplier),
        "diversity_vendi": ("vendi_multiplier", default_multiplier),
    }
    
    calculated_thresholds = {}
    initial_values = {}
    
    for metric_name, (multiplier_key, metric_default) in dynamic_metrics.items():
        if metric_name in rewards:
            initial_value = rewards[metric_name]
            initial_values[metric_name] = initial_value
            
            # Get metric-specific multiplier or use metric-specific default
            multiplier = diversity_thresholds.get(multiplier_key, metric_default)
            
            # Calculate dynamic threshold
            dynamic_threshold = initial_value * multiplier
            calculated_thresholds[metric_name] = dynamic_threshold
            
            logging.info(
                f"[Dynamic Threshold] {metric_name}: initial={initial_value:.4f}, "
                f"multiplier={multiplier:.1f}x, threshold={dynamic_threshold:.4f}"
            )
    
    return calculated_thresholds


def check_early_stopping(
    rewards: Dict[str, float],
    diversity_thresholds: Dict[str, float],
    dynamic_thresholds_calculated: Dict[str, float],
    initial_diversity_values: Dict[str, float],
) -> Tuple[bool, Optional[str]]:
    """
    Check if early stopping conditions are met.
    
    Args:
        rewards: Dictionary of current rewards/diversity scores
        diversity_thresholds: Dictionary containing threshold settings
        dynamic_thresholds_calculated: Pre-calculated dynamic thresholds
        initial_diversity_values: Initial diversity values from iteration 0
    
    Returns:
        Tuple of (should_stop, stop_reason)
    """
    use_dynamic = diversity_thresholds.get('use_dynamic_thresholds', False)
    
    # Map diversity objective names to threshold keys and their dynamic behavior
    # Fixed metrics: LPIPS, DINO, DreamSim - require explicit threshold
    # Dynamic metrics: TinyL2, Color, DPP, Vendi - automatically use dynamic when enabled
    threshold_mapping = {
        "diversity_lpips": ("lpips_threshold", False),         # Fixed only
        "diversity_dino": ("dino_threshold", False),           # Fixed only
        "diversity_dreamsim": ("dreamsim_threshold", False),   # Fixed only
        "diversity_tiny_l2": (None, True),      # Dynamic automatically when enabled
        "diversity_color": (None, True),       # Dynamic automatically when enabled
        "diversity_dpp": (None, True),         # Dynamic automatically when enabled
        "diversity_dpp_patch": (None, True),   # Dynamic automatically when enabled
        "diversity_vendi": (None, True),       # Dynamic automatically when enabled
    }
    
    for diversity_name, (threshold_key, auto_dynamic) in threshold_mapping.items():
        if diversity_name not in rewards:
            continue
        
        current_score = rewards[diversity_name]
        threshold_value = None
        is_dynamic = False
        
        # Dynamic metrics (TinyL2, Color, DPP, Vendi) automatically use dynamic thresholds
        if auto_dynamic and use_dynamic and diversity_name in dynamic_thresholds_calculated:
            # Automatically use dynamic threshold
            threshold_value = dynamic_thresholds_calculated[diversity_name]
            is_dynamic = True
        # Fixed metrics (LPIPS, DINO, DreamSim) require explicit threshold
        elif threshold_key is not None and threshold_key in diversity_thresholds:
            # Use fixed threshold if specified
            threshold_value = diversity_thresholds[threshold_key]
        
        # Check if threshold is met
        if threshold_value is not None and current_score >= threshold_value:
            threshold_type = "dynamic" if is_dynamic else "fixed"
            stop_reason = (
                f"{diversity_name} diversity reached {threshold_type} threshold: "
                f"{current_score:.4f} >= {threshold_value:.4f}"
            )
            if is_dynamic:
                initial_val = initial_diversity_values.get(diversity_name, 0)
                multiplier = threshold_value / initial_val if initial_val > 0 else 0
                stop_reason += f" (initial={initial_val:.4f}, {multiplier:.1f}x)"
            
            logging.info(f"[Early Stopping] {stop_reason}")
            return True, stop_reason
    
    return False, None


def log_early_stopping(iteration: int, stop_reason: str) -> None:
    """
    Log early stopping information.
    
    Args:
        iteration: Iteration number where stopping occurred
        stop_reason: Reason for stopping
    """
    logging.info(f"Stopping optimization at iteration {iteration}: {stop_reason}")


def make_sequential_dpp_stop(
    num_images: int,
    threshold: float = 0.03,
    key: str = "diversity_dpp",
) -> Callable[[Dict[str, float], int], Optional[str]]:
    """
    Build a stop function for sequential DPP sampling that halts when the
    normalized score under `key` reaches (1 - threshold), i.e. 97% of max by
    default. Works for both `diversity_dpp` and `diversity_dpp_patch` since
    both have upper bound num_images * log(2) (BxB cosine kernel logdet).
    """
    if num_images < 1:
        raise ValueError("num_images must be >= 1 for sequential DPP stopping.")
    max_score = num_images * math.log(2.0)
    target_normalized = 1.0 - threshold

    def _stop_fn(rewards: Dict[str, float], iteration: int) -> Optional[str]:
        dpp_value = rewards.get(key)
        if dpp_value is not None:
            normalized = dpp_value / max_score
            if normalized >= target_normalized:
                return (
                    f"{key} reached {target_normalized:.0%} of max: "
                    f"{normalized:.4f} >= {target_normalized:.4f}"
                )
        return None

    return _stop_fn


def make_dpp_absolute_stop(threshold: float) -> Callable[[Dict[str, float], int], Optional[str]]:
    """
    Build a stop function that halts when raw DPP score reaches a fixed threshold.
    """
    def _stop_fn(rewards: Dict[str, float], iteration: int) -> Optional[str]:
        dpp_value = rewards.get("diversity_dpp")
        if dpp_value is not None and dpp_value >= threshold:
            return f"diversity_dpp reached absolute threshold: {dpp_value:.4f} >= {threshold:.4f}"
        return None

    return _stop_fn


def make_dpp_multiplier_stop(multiplier: float) -> Callable[[Dict[str, float], int], Optional[str]]:
    """
    Build a stop function that halts when DPP reaches multiplier * initial_dpp.

    Self-resets on iteration 0 so the same closure can be safely reused
    across multiple `trainer.train()` calls; otherwise `initial_dpp` from a
    previous run would silently determine the target for the next.
    """
    initial_dpp = None

    def _stop_fn(rewards: Dict[str, float], iteration: int) -> Optional[str]:
        nonlocal initial_dpp
        dpp_value = rewards.get("diversity_dpp")
        if dpp_value is None:
            return None

        # Re-anchor on iteration 0 so reusing this closure across runs is safe.
        if iteration == 0 or initial_dpp is None:
            initial_dpp = dpp_value
            logging.info(f"[DPP Multiplier Stop] Initial DPP: {initial_dpp:.4f}, target: {initial_dpp * multiplier:.4f}")
            return None

        target = initial_dpp * multiplier
        if dpp_value >= target:
            return f"diversity_dpp reached {multiplier}x initial: {dpp_value:.4f} >= {target:.4f}"
        return None

    return _stop_fn

