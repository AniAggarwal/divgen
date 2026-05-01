#!/usr/bin/env python3
"""Merge per-prompt results.json files into a single aggregated file."""

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List

# Import LaTeX table printing from analyze_results
# Add parent directory to path to allow importing from utils
import sys
_utils_dir = os.path.dirname(os.path.abspath(__file__))
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)
from analyze_results import print_latex_table


def load_results_from_subdirs(base_dir: str) -> List[Dict]:
    """Load all per-prompt results.json files from prompt subdirectories."""
    results = []
    
    # Find all results.json files in numbered prompt directories
    # Pattern matches: 00000/results.json, 00001/results.json, etc.
    pattern = os.path.join(base_dir, "**/results.json")
    json_files = glob.glob(pattern, recursive=True)
    
    # Filter to only include results.json files in prompt directories (numeric folder names)
    prompt_results = []
    for json_file in json_files:
        # Check if the parent directory name is numeric (prompt index)
        parent_dir = os.path.basename(os.path.dirname(json_file))
        if parent_dir.isdigit():
            prompt_results.append(json_file)
    
    print(f"Found {len(prompt_results)} per-prompt results.json files in {base_dir}")
    
    for json_file in sorted(prompt_results):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = os.path.relpath(json_file, base_dir)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results


def merge_results(results: List[Dict]) -> Dict:
    """Merge per-prompt results into a single aggregated dictionary."""
    if not results:
        return {}
    
    # Each result is already a per-prompt result
    all_prompts = results
    
    # Sort by prompt index
    all_prompts.sort(key=lambda x: x.get("prompt_index", 0))
    
    # Compute overall statistics
    num_prompts = len(all_prompts)
    
    if num_prompts == 0:
        return {"error": "No prompts found"}
    
    # Initialize accumulators
    first_prompt = all_prompts[0]
    metric_keys = list(first_prompt.get("initial_rewards", {}).keys())
    initial_values = {k: [] for k in metric_keys}
    best_values = {k: [] for k in metric_keys}
    
    if not metric_keys:
        return {"error": "No metrics found in results"}
    
    # Accumulate rewards
    for prompt in all_prompts:
        initial_rewards = prompt.get("initial_rewards", {})
        best_rewards = prompt.get("best_rewards", {})
        for metric_key in metric_keys:
            initial_values[metric_key].append(initial_rewards.get(metric_key, 0.0))
            best_values[metric_key].append(best_rewards.get(metric_key, 0.0))
    
    # Compute means
    mean_initial = {
        k: sum(values) / num_prompts for k, values in initial_values.items()
    }
    mean_best = {k: sum(values) / num_prompts for k, values in best_values.items()}
    
    # Compute population standard deviations (zero when only one sample)
    std_initial = {
        k: statistics.pstdev(values) if values else 0.0
        for k, values in initial_values.items()
    }
    std_best = {
        k: statistics.pstdev(values) if values else 0.0
        for k, values in best_values.items()
    }
    
    # Calculate iteration statistics if available
    iterations = [p.get("num_iterations") for p in all_prompts if "num_iterations" in p]
    iteration_stats = None
    if iterations:
        iteration_stats = {
            "mean": sum(iterations) / len(iterations),
            "min": min(iterations),
            "max": max(iterations),
            "total_with_data": len(iterations),
        }
    
    # Get min/max prompt indices
    prompt_indices = [p.get("prompt_index", 0) for p in all_prompts]
    min_idx = min(prompt_indices) if prompt_indices else 0
    max_idx = max(prompt_indices) if prompt_indices else 0
    
    # Check if there's a subset (T2I-CompBench)
    subsets = set(p.get("subset") for p in all_prompts if "subset" in p)
    subset_info = list(subsets)[0] if len(subsets) == 1 else None
    
    # Create merged output
    merged = {
        "summary": {
            "num_prompts": num_prompts,
            "total_prompts": num_prompts,  # Keep for backward compatibility
            "num_sources": len(results),
            "start_index": min_idx,
            "end_index": max_idx,
            "prompt_index_range": [min_idx, max_idx],  # Keep for backward compatibility
            "mean_initial_rewards": mean_initial,
            "std_initial_rewards": std_initial,
            "mean_best_rewards": mean_best,
            "std_best_rewards": std_best,
        }
    }
    
    if iteration_stats:
        merged["summary"]["iteration_stats"] = iteration_stats
    
    if subset_info:
        merged["summary"]["subset"] = subset_info
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge results.json files from multiple GPUs"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory containing subdirectories with results.json files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: merged_results.json in the output_dir)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for LaTeX table (default: derived from output directory name)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist")
        sys.exit(1)
    
    # Load all results
    results = load_results_from_subdirs(args.output_dir)
    
    if not results:
        print("No results.json files found!")
        sys.exit(1)
    
    # Merge results
    print("Merging results...")
    merged = merge_results(results)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.output_dir, "merged_results.json")
    
    # Save merged results
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✓ Merged {len(results)} result files → {output_path}")
    print(f"  Total prompts: {merged['summary']['num_prompts']}")
    if "start_index" in merged["summary"]:
        print(f"  Prompt range: {merged['summary']['start_index']}-{merged['summary']['end_index']}")
    
    # Always print LaTeX table
    # Determine experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        # Derive from output directory name
        exp_name = os.path.basename(os.path.abspath(args.output_dir))
        if not exp_name:
            exp_name = "Experiment"
    
    print_latex_table(merged, exp_name)


if __name__ == "__main__":
    main()

