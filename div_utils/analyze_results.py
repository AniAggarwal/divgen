#!/usr/bin/env python3
"""Analyze merged results and generate LaTeX tables."""

import json
import sys
import argparse
from typing import Dict
import numpy as np


def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_latex_table(results: Dict, experiment_name: str = "Experiment"):
    """Print LaTeX table rows with mean ± std for initial and best rewards."""
    summary = results.get("summary", {})
    per_prompt = results.get("per_prompt_results", [])
    use_summary_stats = (
        summary.get("mean_initial_rewards")
        and summary.get("mean_best_rewards")
        and summary.get("std_initial_rewards")
        and summary.get("std_best_rewards")
    )

    def normalize_metric_name(metric: str) -> str:
        """Normalize metric name to canonical form (lowercase, handle diversity_X vs X_diversity)."""
        metric_lower = metric.lower()
        
        # Handle diversity metrics: normalize diversity_X and X_diversity to diversity_X
        if metric_lower.startswith("diversity_"):
            return metric_lower
        elif metric_lower.endswith("_diversity"):
            # Convert X_diversity to diversity_X
            base = metric_lower[:-10]  # Remove "_diversity"
            return f"diversity_{base}"
        elif "_diversity_" in metric_lower:
            # Handle cases like vendi_sscd_diversity -> diversity_vendi_sscd
            parts = metric_lower.split("_diversity_")
            if len(parts) == 2:
                return f"diversity_{parts[0]}_{parts[1]}"
            return metric_lower
        else:
            # Reward metrics: just lowercase
            return metric_lower
    
    # Define metric display mapping (canonical name -> LaTeX display name)
    metric_display_names = {
        # Diversity metrics (canonical form: diversity_X)
        "diversity_dino": "DINO",
        "diversity_dreamsim": "DreamSim",
        "diversity_lpips": "LPIPS",
        "diversity_color": "Color",
        "diversity_tiny_l2": "Tiny L2",
        "diversity_dpp": "DPP",
        "diversity_vendi": "Vendi",
        "diversity_vendi_sscd": "Vendi-SSCD",
        # Reward metrics
        "hps": "HPSv2",
        "clip": "CLIPScore",
        "clip_b32": "CLIP-B32",
        "imagereward": "ImageReward",
        "pickscore": "PickScore",
    }
    
    # Collect metrics from summary if available, otherwise fall back to per-prompt data
    if use_summary_stats:
        mean_initial = summary.get("mean_initial_rewards", {})
        mean_best = summary.get("mean_best_rewards", {})
        available_metrics = list(set(mean_initial.keys()) | set(mean_best.keys()))
    elif per_prompt:
        first_prompt = per_prompt[0]
        available_metrics = list(first_prompt.get("initial_rewards", {}).keys())
    else:
        print("\nNo data available for LaTeX table.")
        return
    
    # Normalize and filter metrics
    # Use a dict to map normalized -> original, keeping first occurrence
    normalized_to_original = {}
    display_metrics = []
    seen_normalized = set()
    for metric in available_metrics:
        normalized = normalize_metric_name(metric)
        if normalized in metric_display_names and normalized not in seen_normalized:
            normalized_to_original[normalized] = metric
            display_metrics.append(normalized)
            seen_normalized.add(normalized)
    
    if not display_metrics:
        print("\nNo displayable metrics found for LaTeX table.")
        return
    
    # Collect mean/std data for each metric
    initial_stats = {}
    best_stats = {}

    if use_summary_stats:
        std_initial = summary.get("std_initial_rewards", {})
        std_best = summary.get("std_best_rewards", {})
        for normalized_metric in display_metrics:
            original_metric = normalized_to_original[normalized_metric]
            if original_metric in mean_initial and original_metric in std_initial:
                mean_val = mean_initial[original_metric]
                std_val = std_initial[original_metric]
                if normalized_metric == "hps":
                    mean_val = 1.0 - mean_val
                initial_stats[normalized_metric] = {
                    "mean": mean_val,
                    "std": std_val,
                }
            if original_metric in mean_best and original_metric in std_best:
                mean_val = mean_best[original_metric]
                std_val = std_best[original_metric]
                if normalized_metric == "hps":
                    mean_val = 1.0 - mean_val
                best_stats[normalized_metric] = {
                    "mean": mean_val,
                    "std": std_val,
                }
    else:
        initial_data = {m: [] for m in display_metrics}
        best_data = {m: [] for m in display_metrics}
        
        for result in per_prompt:
            for normalized_metric in display_metrics:
                original_metric = normalized_to_original[normalized_metric]
                init_val = result.get("initial_rewards", {}).get(original_metric)
                best_val = result.get("best_rewards", {}).get(original_metric)
                
                # Transform HPS: show as 1 - x (since HPS is a loss where lower is better)
                if normalized_metric == "hps":
                    if init_val is not None and not np.isnan(init_val):
                        initial_data[normalized_metric].append(1.0 - init_val)
                    if best_val is not None and not np.isnan(best_val):
                        best_data[normalized_metric].append(1.0 - best_val)
                else:
                    if init_val is not None and not np.isnan(init_val):
                        initial_data[normalized_metric].append(init_val)
                    if best_val is not None and not np.isnan(best_val):
                        best_data[normalized_metric].append(best_val)
        
        for metric in display_metrics:
            if initial_data[metric]:
                initial_stats[metric] = {
                    "mean": np.mean(initial_data[metric]),
                    "std": np.std(initial_data[metric])
                }
            if best_data[metric]:
                best_stats[metric] = {
                    "mean": np.mean(best_data[metric]),
                    "std": np.std(best_data[metric])
                }
    
    print("\n" + "="*70)
    print("LATEX TABLE OUTPUT")
    print("="*70)
    
    # Print header row
    header_parts = ["\\textbf{Objective}"]
    for normalized_metric in display_metrics:
        display_name = metric_display_names[normalized_metric]
        header_parts.append(display_name)
    
    print("\n" + " & ".join(header_parts) + " \\\\")
    print("\\midrule")
    
    # Print initial row
    init_parts = [f"{experiment_name} init"]
    for normalized_metric in display_metrics:
        if normalized_metric in initial_stats:
            mean = initial_stats[normalized_metric]["mean"]
            std = initial_stats[normalized_metric]["std"]
            init_parts.append(f"{mean:.3f}$_{{\\pm {std:.3f}}}$")
        else:
            init_parts.append("N/A")
    
    print(" & ".join(init_parts) + " \\\\")
    
    # Print best row
    best_parts = [f"{experiment_name} best"]
    for normalized_metric in display_metrics:
        if normalized_metric in best_stats:
            mean = best_stats[normalized_metric]["mean"]
            std = best_stats[normalized_metric]["std"]
            best_parts.append(f"{mean:.3f}$_{{\\pm {std:.3f}}}$")
        else:
            best_parts.append("N/A")
    
    print(" & ".join(best_parts) + " \\\\")
    print()


def print_summary_stats(results: Dict):
    """Print summary statistics."""
    summary = results.get("summary", {})
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal prompts: {summary.get('num_prompts', 'N/A')}")
    if "start_index" in summary:
        print(f"Prompt range: {summary['start_index']} to {summary['end_index']}")
    if "subset" in summary:
        print(f"Subset: {summary['subset']}")
    if "num_sources" in summary:
        print(f"Number of source files merged: {summary['num_sources']}")
    
    # Iteration statistics if available
    if "iteration_stats" in summary:
        stats = summary["iteration_stats"]
        print(f"\nIteration Statistics:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
    
    # Mean metrics
    print("\n" + "-"*70)
    print("MEAN INITIAL REWARDS")
    print("-"*70)
    for metric, value in summary.get("mean_initial_rewards", {}).items():
        print(f"  {metric:30s}: {value:10.4f}")
    
    print("\n" + "-"*70)
    print("MEAN BEST REWARDS")
    print("-"*70)
    for metric, value in summary.get("mean_best_rewards", {}).items():
        print(f"  {metric:30s}: {value:10.4f}")
    


def main():
    parser = argparse.ArgumentParser(
        description="Analyze results from JSON file"
    )
    parser.add_argument(
        "results_json",
        type=str,
        help="Path to results.json or merged_results.json"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for LaTeX table output (default: derived from filename)"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="(deprecated) explicitly request LaTeX table output"
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip LaTeX table output"
    )
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_results(args.results_json)
    except FileNotFoundError:
        print(f"Error: File not found: {args.results_json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    
    # Print summary statistics
    print_summary_stats(results)
    
    # Always print LaTeX table unless explicitly suppressed
    should_print_latex = not args.no_latex or args.latex
    if should_print_latex:
        if args.experiment_name:
            exp_name = args.experiment_name
        else:
            # Try to derive from filename
            import os
            dirname = os.path.dirname(args.results_json)
            exp_name = os.path.basename(dirname) if dirname else "Experiment"
        
        print_latex_table(results, exp_name)
    


if __name__ == "__main__":
    main()

