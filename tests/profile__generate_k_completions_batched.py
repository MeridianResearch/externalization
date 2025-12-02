#!/usr/bin/env python3
"""
Profile and compare performance of generate_k_completions vs generate_k_completions_batched.

This script supports two modes:
1. Comparison mode (default): Compare sequential vs batched performance
2. Batch-only mode: Optimize hyperparameters (k, num_prompts) for batched version

The script outputs verbose progress information plus an ASCII table summary of results.

Usage Examples:
--------------

# Comparison mode (default)
python tests/profile__generate_k_completions_batched.py

# Comparison mode with custom parameters
python tests/profile__generate_k_completions_batched.py --num_prompts 2 --k 8 --max_new_tokens 512

# Batch-only mode with default grid (k: 2,4,8 | prompts: 1,2,4)
python tests/profile__generate_k_completions_batched.py --mode batch-only

# Custom hyperparameter search
python tests/profile__generate_k_completions_batched.py --mode batch-only \
    --k_values "2,4,8,16" \
    --prompt_values "1,2,4,8" \
    --metric throughput

# Find optimal configuration for maximum tokens/sec
python tests/profile__generate_k_completions_batched.py --mode batch-only \
    --k_values "4,8,16,32" \
    --prompt_values "1,2,4" \
    --metric tokens_per_sec \
    --max_new_tokens 512

# Quick test with base model only (skip SFT checkpoint)
python tests/profile__generate_k_completions_batched.py --mode batch-only \
    --skip_model_load \
    --max_new_tokens 32 \
    --k_values "2,4" \
    --prompt_values "1,2"
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np

import torch

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from early_exit.util import get_model, load_model_from_wandb, load_model
from early_exit.rl_utils import generate_k_completions, generate_k_completions_batched
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from early_exit.rl_types import RLHyperparams
from shared_utils.load import get_tokenizer, configs_from_yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile generate_k_completions vs generate_k_completions_batched"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["comparison", "batch-only"],
        default="comparison",
        help="Profiling mode (default: comparison)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1,
        help="Number of prompts to test with (default: 1) [comparison mode only]",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of completions per prompt (default: 4) [comparison mode only]",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=700,
        help="Maximum number of new tokens to generate (default: 700)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of timed runs (default: 3)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--skip_model_load",
        action="store_true",
        help="Skip loading the SFT checkpoint (use base model only)",
    )
    # Batch-only mode arguments
    parser.add_argument(
        "--k_values",
        type=str,
        default="2,4,8",
        help="Comma-separated k values to test in batch-only mode (default: 2,4,8)",
    )
    parser.add_argument(
        "--prompt_values",
        type=str,
        default="1,2,4",
        help="Comma-separated prompt counts to test in batch-only mode (default: 1,2,4)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "tokens_per_sec"],
        default="time",
        help="Optimization metric for batch-only mode (default: time)",
    )
    return parser.parse_args()


def setup_model(device, skip_model_load=False):
    """
    Setup the model matching the notebook configuration.
    
    This includes:
    - Loading the base model
    - Applying replace_attention_layers
    - Loading SFT checkpoint weights
    """
    print("=" * 80)
    print("MODEL SETUP")
    print("=" * 80)
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config_path = ROOT_DIR / "config_deepseek.yaml"
    sft_model_path = ROOT_DIR / "models/early_exit_20250908_layers_5_big"
    
    print(f"Model name: {model_name}")
    print(f"Config path: {config_path}")
    print(f"Device: {device}")
    
    # Load tokenizer and config
    print("\n[1/4] Loading tokenizer and config...")
    tokenizer = get_tokenizer(model_name)
    config = configs_from_yaml(config_path, tokenizer.eos_token_id)
    
    # Load base model
    print(f"[2/4] Loading base model...")
    student = get_model(model_name, config['model'], device)
    
    # Apply attention layer replacement (critical for early exit)
    print(f"[3/4] Applying replace_attention_layers...")
    student = replace_attention_layers(student, config['lora'], device)
    
    # Load SFT checkpoint
    if not skip_model_load:
        print(f"[4/4] Loading SFT checkpoint from {sft_model_path}...")
        student = load_model_from_wandb(
            student,
            model_path=sft_model_path,
            artifact_path='vkarthik095-university-of-amsterdam/early-exit/early_exit_20250908_layers_5_big:v0'
        )
        student = load_model(student, sft_model_path)
    else:
        print(f"[4/4] Skipping SFT checkpoint load (using base model only)")
    
    # Set to eval mode
    student.eval()
    
    print("\n✓ Model setup complete")
    print("=" * 80)
    
    return student, tokenizer, config


def get_test_prompts(num_prompts):
    """Get test prompts for profiling."""
    all_prompts = [
        "Who is the president of Burundi",
        "What is the capital of Rwanda",
        "How many continents are there",
        "What is 15 multiplied by 23",
        "Solve for x: 2x + 5 = 17",
        "What is the square root of 144",
        "How many days are in a leap year",
        "What is the largest planet in our solar system",
    ]
    return all_prompts[:num_prompts]


def profile_function(func_name, func, prompts, k, tokenizer, config, device, system_prompt, num_runs, warmup_runs):
    """
    Profile a generation function.
    
    Returns:
        times (list): List of execution times for each run
        completions: Last run's completions
        exit_info: Last run's exit info
    """
    print(f"\n{'=' * 80}")
    print(f"PROFILING: {func_name}")
    print(f"{'=' * 80}")
    print(f"Prompts: {len(prompts)}")
    print(f"K completions per prompt: {k}")
    print(f"Total completions: {len(prompts) * k}")
    
    times = []
    completions = None
    exit_info = None
    
    # Warmup runs
    if warmup_runs > 0:
        print(f"\n--- Warmup Phase ({warmup_runs} run{'s' if warmup_runs > 1 else ''}) ---")
        for i in range(warmup_runs):
            print(f"Warmup run {i+1}/{warmup_runs}...", end=" ", flush=True)
            with torch.no_grad():
                if device == "cuda":
                    torch.cuda.synchronize()
                
                _ = func(
                    model=None,  # Will be filled by the actual call
                    prompts=prompts,
                    k=k,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    system_prompt=system_prompt
                )
                
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            print("✓")
    
    # Timed runs
    print(f"\n--- Timed Runs ({num_runs} run{'s' if num_runs > 1 else ''}) ---")
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
        
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            completions, exit_info = func(
                model=None,  # Will be filled by the actual call
                prompts=prompts,
                k=k,
                tokenizer=tokenizer,
                config=config,
                device=device,
                system_prompt=system_prompt
            )
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            if device == "cuda":
                torch.cuda.empty_cache()
        
        print(f"✓ ({elapsed_time:.3f}s)")
    
    return times, completions, exit_info


def print_summary_table(sequential_times, batched_times):
    """Print ASCII table summary of results."""
    # Calculate statistics
    seq_mean = np.mean(sequential_times)
    seq_std = np.std(sequential_times)
    seq_min = np.min(sequential_times)
    seq_max = np.max(sequential_times)
    
    batch_mean = np.mean(batched_times)
    batch_std = np.std(batched_times)
    batch_min = np.min(batched_times)
    batch_max = np.max(batched_times)
    
    speedup = seq_mean / batch_mean if batch_mean > 0 else 0
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # ASCII table
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "Performance Comparison Summary" + " " * 23 + "║")
    print("╠" + "═" * 78 + "╣")
    print("║ Function                        │ Mean (s) │  Std  │  Min  │  Max   ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ generate_k_completions          │  {seq_mean:6.3f}  │ {seq_std:5.3f} │ {seq_min:5.3f} │ {seq_max:6.3f} ║")
    print(f"║ generate_k_completions_batched  │  {batch_mean:6.3f}  │ {batch_std:5.3f} │ {batch_min:5.3f} │ {batch_max:6.3f} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ Speedup: {speedup:.2f}x faster (batched vs sequential)" + " " * (78 - 48 - len(f"{speedup:.2f}")) + "║")
    print("╚" + "═" * 78 + "╝")
    print()


def print_batch_only_results(results, metric):
    """Print ASCII table summary of hyperparameter search results."""
    # Sort by optimization metric
    if metric == "time":
        results_sorted = sorted(results, key=lambda x: x['mean_time'])
        best = results_sorted[0]
    elif metric == "throughput":
        results_sorted = sorted(results, key=lambda x: x['throughput'], reverse=True)
        best = results_sorted[0]
    elif metric == "tokens_per_sec":
        results_sorted = sorted(results, key=lambda x: x['tokens_per_sec'], reverse=True)
        best = results_sorted[0]
    
    print("\n" + "=" * 90)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 90)
    
    # Print table with all configurations
    print()
    print("╔" + "═" * 88 + "╗")
    print("║" + " " * 28 + "Hyperparameter Search Results" + " " * 31 + "║")
    print("╠" + "═" * 88 + "╣")
    print("║ Prompts │   K   │ Time (s) │ Throughput │ Tokens/sec │  Std  │  Status    ║")
    print("╠" + "═" * 88 + "╣")
    
    for result in results_sorted:
        status = "★ OPTIMAL" if result == best else ""
        std = np.std(result['times'])
        print(f"║   {result['num_prompts']:2d}    │  {result['k']:3d}  │  {result['mean_time']:6.3f}  │   {result['throughput']:7.2f}   │   {result['tokens_per_sec']:8.1f}  │ {std:5.3f} │ {status:10s} ║")
    
    print("╚" + "═" * 88 + "╝")
    
    # Print optimal configuration details
    print(f"\n🎯 OPTIMAL CONFIGURATION (optimized for {metric}):")
    print(f"   Prompts: {best['num_prompts']}")
    print(f"   K: {best['k']}")
    print(f"   Mean time: {best['mean_time']:.3f}s")
    print(f"   Throughput: {best['throughput']:.2f} completions/sec")
    print(f"   Tokens per second: {best['tokens_per_sec']:.1f}")
    print()


def run_comparison_mode(args):
    """Run comparison mode: profile both sequential and batched versions."""
    print("\n" + "=" * 80)
    print("COMPARISON MODE: Sequential vs Batched")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of prompts: {args.num_prompts}")
    print(f"  - K completions per prompt: {args.k}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Number of timed runs: {args.num_runs}")
    print(f"  - Number of warmup runs: {args.warmup_runs}")
    print(f"  - Device: {args.device}")
    print("=" * 80)
    
    # Setup model
    student, tokenizer, config = setup_model(args.device, args.skip_model_load)
    
    # Configure generation
    config['generation']['max_new_tokens'] = args.max_new_tokens
    
    # Get test prompts
    prompts = get_test_prompts(args.num_prompts)
    print(f"\nTest prompts ({len(prompts)}):")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")
    
    # System prompt
    RL_HPARAMS = RLHyperparams()
    system_prompt = RL_HPARAMS.system_prompt
    
    # Create wrapper functions that include the model
    def generate_k_completions_wrapper(model, prompts, k, tokenizer, config, device, system_prompt):
        return generate_k_completions(student, prompts, k, tokenizer, config, device, system_prompt)
    
    def generate_k_completions_batched_wrapper(model, prompts, k, tokenizer, config, device, system_prompt):
        return generate_k_completions_batched(student, prompts, k, tokenizer, config, device, system_prompt)
    
    # Profile sequential version
    sequential_times, seq_completions, seq_exit_info = profile_function(
        func_name="generate_k_completions (sequential)",
        func=generate_k_completions_wrapper,
        prompts=prompts,
        k=args.k,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
        system_prompt=system_prompt,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Profile batched version
    batched_times, batch_completions, batch_exit_info = profile_function(
        func_name="generate_k_completions_batched",
        func=generate_k_completions_batched_wrapper,
        prompts=prompts,
        k=args.k,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
        system_prompt=system_prompt,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )
    
    # Print summary table
    print_summary_table(sequential_times, batched_times)
    
    # Print sample output info
    print("Sample Output Info:")
    print(f"  - Sequential: {len(seq_completions['texts'])} completions generated")
    print(f"  - Batched: {len(batch_completions['texts'])} completions generated")
    print(f"  - Token tensor shape (sequential): {seq_completions['tokens'].shape}")
    print(f"  - Token tensor shape (batched): {batch_completions['tokens'].shape}")
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80 + "\n")


def run_batch_only_mode(args):
    """Run batch-only mode: hyperparameter search for batched version."""
    # Parse k_values and prompt_values from CSV strings
    k_values = [int(x.strip()) for x in args.k_values.split(',')]
    prompt_values = [int(x.strip()) for x in args.prompt_values.split(',')]
    
    print("\n" + "=" * 80)
    print("BATCH-ONLY MODE: Hyperparameter Search")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - K values to test: {k_values}")
    print(f"  - Prompt values to test: {prompt_values}")
    print(f"  - Total configurations: {len(k_values) * len(prompt_values)}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Number of timed runs per config: {args.num_runs}")
    print(f"  - Number of warmup runs per config: {args.warmup_runs}")
    print(f"  - Optimization metric: {args.metric}")
    print(f"  - Device: {args.device}")
    print("=" * 80)
    
    # Setup model once
    student, tokenizer, config = setup_model(args.device, args.skip_model_load)
    config['generation']['max_new_tokens'] = args.max_new_tokens
    
    # System prompt
    RL_HPARAMS = RLHyperparams()
    system_prompt = RL_HPARAMS.system_prompt
    
    # Create wrapper function that includes the model
    def generate_k_completions_batched_wrapper(model, prompts, k, tokenizer, config, device, system_prompt):
        return generate_k_completions_batched(student, prompts, k, tokenizer, config, device, system_prompt)
    
    # Grid search over all combinations
    results = []
    total_configs = len(k_values) * len(prompt_values)
    config_num = 0
    
    for num_prompts in prompt_values:
        for k in k_values:
            config_num += 1
            prompts = get_test_prompts(num_prompts)
            
            print(f"\n{'=' * 80}")
            print(f"Testing configuration {config_num}/{total_configs}: prompts={num_prompts}, k={k}")
            print(f"{'=' * 80}")
            
            times, completions, _ = profile_function(
                func_name=f"generate_k_completions_batched (prompts={num_prompts}, k={k})",
                func=generate_k_completions_batched_wrapper,
                prompts=prompts,
                k=k,
                tokenizer=tokenizer,
                config=config,
                device=args.device,
                system_prompt=system_prompt,
                num_runs=args.num_runs,
                warmup_runs=args.warmup_runs
            )
            
            # Calculate metrics
            mean_time = np.mean(times)
            total_completions = num_prompts * k
            throughput = total_completions / mean_time if mean_time > 0 else 0
            
            # Estimate tokens per second
            total_tokens = completions['tokens'].numel()
            tokens_per_sec = total_tokens / mean_time if mean_time > 0 else 0
            
            results.append({
                'num_prompts': num_prompts,
                'k': k,
                'mean_time': mean_time,
                'throughput': throughput,
                'tokens_per_sec': tokens_per_sec,
                'times': times
            })
            
            print(f"  → Mean time: {mean_time:.3f}s")
            print(f"  → Throughput: {throughput:.2f} completions/sec")
            print(f"  → Tokens/sec: {tokens_per_sec:.1f}")
    
    # Print results table
    print_batch_only_results(results, args.metric)
    
    print("=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80 + "\n")


def main():
    """Main profiling script."""
    args = parse_args()
    
    if args.mode == "comparison":
        run_comparison_mode(args)
    elif args.mode == "batch-only":
        run_batch_only_mode(args)


if __name__ == "__main__":
    main()

