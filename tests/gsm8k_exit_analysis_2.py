import torch
import sys
import os
import argparse
import gzip
import pickle
from typing import List, Dict, Any
import pandas as pd
from collections import defaultdict
from huggingface_hub import hf_hub_download
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

from tests.style import orange_gradient, TITLE_FONTSIZE, LABEL_FONTSIZE, TICK_FONTSIZE, ANNOTATION_FONTSIZE


def load_gsm8k_with_difficulty():
    """Load the GSM8K dataset with teacher exit probabilities"""
    import gzip
    import pickle
    
    enriched_file = hf_hub_download(
        repo_id="lizardp1/gsm8k_early_exit",
        filename="sft_correct_only.pkl.gz",
        repo_type="dataset"
    )
    
    # Load all samples from the gzipped pickle
    samples = []
    with gzip.open(enriched_file, "rb") as f:
        header = pickle.load(f)  # {'metadata': ...}
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict) and obj.get('_end'):
                break
            samples.append(obj)
    
    return samples


def calculate_teacher_exit_probs(teacher_sample, exitable_layer_idxs, device='cpu'):

    kl_div_per_layer = teacher_sample['kl_div1_per_layer'].to(device)
    
    current_exitable_indices = exitable_layer_idxs[:-1].long()
    kl_div_per_layer = kl_div_per_layer[:, current_exitable_indices, :]
    
    KL_FACTOR = 1.0
    
    sigmoid_kls = torch.sigmoid(KL_FACTOR * kl_div_per_layer)  # [batch, num_layers, seq_len]
    sigmoid_kls = 2.0 * sigmoid_kls - 1.0
    sigmoid_kls = 1.0 - sigmoid_kls
    
    batch_size, num_layers, seq_len = sigmoid_kls.shape
    stickbreaking_probs = torch.zeros(batch_size, num_layers + 1, seq_len, device=sigmoid_kls.device)
    
    for l in range(num_layers):
        if l == 0:
            prod_term = torch.ones((batch_size, seq_len), device=sigmoid_kls.device)
        else:
            prod_term = torch.prod(1 - sigmoid_kls[:, :l, :], dim=1)
        stickbreaking_probs[:, l, :] = sigmoid_kls[:, l, :] * prod_term
    
    stickbreaking_probs[:, -1, :] = torch.prod(1 - sigmoid_kls, dim=1)
    
    early_exit_probs = stickbreaking_probs.permute(0, 2, 1)  # [batch, seq_len, num_layers+1]
    
    return early_exit_probs


def sample_balanced_by_difficulty(samples, max_samples=100, random_seed=5):
    import random
    random.seed(random_seed)

    easy_examples = []
    medium_examples = []
    hard_examples = []
    
    for sample in samples:
        difficulty = sample.get('difficulty_category', None)
        
        if difficulty is None:
            if len(easy_examples) <= len(medium_examples) and len(easy_examples) <= len(hard_examples):
                difficulty = 'Easy'
            elif len(medium_examples) <= len(hard_examples):
                difficulty = 'Medium'
            else:
                difficulty = 'Hard'
        
        if difficulty == 'Easy':
            easy_examples.append(sample)
        elif difficulty == 'Medium':
            medium_examples.append(sample)
        elif difficulty == 'Hard':
            hard_examples.append(sample)
    
    samples_per_category = max_samples // 3
    remainder = max_samples % 3
    
    random.shuffle(easy_examples)
    random.shuffle(medium_examples)
    random.shuffle(hard_examples)
    
    selected_examples = []
    selected_examples.extend(easy_examples[:samples_per_category + (1 if remainder > 0 else 0)])
    selected_examples.extend(medium_examples[:samples_per_category + (1 if remainder > 1 else 0)])
    selected_examples.extend(hard_examples[:samples_per_category])
    
    random.shuffle(selected_examples)
    
    return selected_examples


def plot_exit_distribution_overall(results: List[Dict], teacher_probs_overall, 
                                   exitable_layer_idxs, save_path: str = 'exit_distribution_overall.png'):
    exit_data = defaultdict(int)
    total_tokens = 0
    
    for result in results:
        num_tokens = result['num_tokens']
        layer_dist = eval(result['layer_distribution']) if isinstance(result['layer_distribution'], str) else result['layer_distribution']
        
        total_tokens += num_tokens
        
        for layer, count in layer_dist.items():
            exit_data[layer] += count
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    layers = sorted(exit_data.keys())
    counts = [exit_data[layer] for layer in layers]
    early_exits = sum(counts)
    
    final_layer_pos = max(layers) + 5 if layers else 30
    final_layer_count = total_tokens - early_exits
    layers.append(final_layer_pos)
    counts.append(final_layer_count)
    
    percentages = [(count / total_tokens * 100) if total_tokens > 0 else 0 for count in counts]
    
    exitable_layers_list = exitable_layer_idxs[:-1].tolist()  # Exclude final inf
    teacher_x_positions = exitable_layers_list + [final_layer_pos]
    
    colors = orange_gradient(len(layers))
    
    teacher_percentages = teacher_probs_overall * 100  # conv to percentage
    
    # Bar width and offset for side-by-side bars
    bar_width = 1.8
    offset = bar_width / 2
    
    # Teacher bars (left)
    teacher_bars = ax.bar([x - offset for x in teacher_x_positions], teacher_percentages, 
           width=bar_width, color='gray', edgecolor='black', linewidth=1.5,
           alpha=0.6, zorder=2, label='Teacher Target')
    
    # Actual bars (right) - with outline for visibility
    actual_bars = ax.bar([x + offset for x in layers], percentages, 
           width=bar_width, color=colors, edgecolor='black', linewidth=1.2, zorder=2)
    
    max_height = max([max(percentages), max(teacher_percentages)])
    
    # Add labels on top of teacher bars
    for x_pos, percentage in zip(teacher_x_positions, teacher_percentages):
        if percentage > 0.5:
            ax.text(x_pos - offset, percentage + max_height * 0.02, 
                    f'{percentage:.1f}%', 
                    ha='center', va='bottom', 
                    fontsize=ANNOTATION_FONTSIZE, 
                    fontweight='bold', zorder=4)
    
    # Add labels on top of actual bars
    for x_pos, percentage in zip(layers, percentages):
        if percentage > 0.5:
            ax.text(x_pos + offset, percentage + max_height * 0.02, 
                    f'{percentage:.1f}%', 
                    ha='center', va='bottom', 
                    fontsize=ANNOTATION_FONTSIZE, 
                    fontweight='bold', zorder=4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Percentage of Tokens (%)', fontsize=LABEL_FONTSIZE, fontweight='bold')
    ax.set_xlabel('Exit Layer', fontsize=LABEL_FONTSIZE, fontweight='bold')
    
    exit_rate = early_exits / total_tokens if total_tokens > 0 else 0
    
    ax.set_title(f'Distribution of Exit Layers', 
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(False)
    ax.legend(loc='upper left', fontsize=ANNOTATION_FONTSIZE, framealpha=0.9)
    all_x_layers = sorted(list(exit_data.keys())) + [final_layer_pos]
    ax.set_xticks(all_x_layers)
    labels = [str(layer) for layer in all_x_layers[:-1]] + ['Final Layer']
    ax.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_exit_distribution(results: List[Dict], teacher_probs_by_difficulty: Dict, 
                          exitable_layer_idxs, save_path: str = 'exit_distribution.png'):

    difficulty_data = {
        'Easy': defaultdict(int),
        'Medium': defaultdict(int),
        'Hard': defaultdict(int)
    }
    difficulty_totals = {'Easy': 0, 'Medium': 0, 'Hard': 0}
    
    for result in results:
        difficulty = result['difficulty']
        num_tokens = result['num_tokens']
        layer_dist = eval(result['layer_distribution']) if isinstance(result['layer_distribution'], str) else result['layer_distribution']
        
        difficulty_totals[difficulty] += num_tokens
        
        for layer, count in layer_dist.items():
            difficulty_data[difficulty][layer] += count
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    difficulties = ['Easy', 'Medium', 'Hard']
    
    all_layers = set()
    for diff_data in difficulty_data.values():
        all_layers.update(diff_data.keys())
    all_layers = sorted(list(all_layers))
    
    final_layer_pos = max(all_layers) + 5 if all_layers else 30

    exitable_layers_list = exitable_layer_idxs[:-1].tolist()  # Exclude final inf
    teacher_x_positions = exitable_layers_list + [final_layer_pos]
    
    for idx, (ax, difficulty) in enumerate(zip(axes, difficulties)):
        exit_data = difficulty_data[difficulty]
        total_tokens = difficulty_totals[difficulty]
        
        if total_tokens == 0:
            continue
        
        layers = sorted(exit_data.keys())
        counts = [exit_data[layer] for layer in layers]
        
        early_exits = sum(counts)
        
        final_layer_count = total_tokens - early_exits
        layers.append(final_layer_pos)
        counts.append(final_layer_count)
        
        percentages = [(count / total_tokens * 100) if total_tokens > 0 else 0 for count in counts]
        
        colors = orange_gradient(len(layers))
        
        # Bar width and offset for side-by-side bars
        bar_width = 0.75
        offset = bar_width / 2
        
        if difficulty in teacher_probs_by_difficulty:
            teacher_probs = teacher_probs_by_difficulty[difficulty]  # [num_layers+1]
            teacher_percentages = teacher_probs * 100
            
            # Teacher bars (left)
            ax.bar([x - offset for x in teacher_x_positions], teacher_percentages, 
                   width=bar_width, color='gray', edgecolor='black', linewidth=1.5,
                   alpha=0.6, zorder=2, label='Teacher Target')
            
            # Add labels on teacher bars
            max_val = max([max(percentages), max(teacher_percentages)])
            for x_pos, percentage in zip(teacher_x_positions, teacher_percentages):
                if percentage > 0.5:
                    ax.text(x_pos - offset, percentage + max_val * 0.02, 
                            f'{percentage:.1f}%', 
                            ha='center', va='bottom', 
                            fontsize=ANNOTATION_FONTSIZE, 
                            fontweight='bold', zorder=4)
        
        # Actual bars (right) - with outline for visibility
        bars = ax.bar([x + offset for x in layers], percentages, 
                      width=bar_width, color=colors, edgecolor='black', linewidth=1.2, zorder=2)
        
        # Add labels on actual bars
        max_val = max([max(percentages), max(teacher_probs_by_difficulty[difficulty] * 100) if difficulty in teacher_probs_by_difficulty else 0])
        for x_pos, percentage in zip(layers, percentages):
            if percentage > 0.5:  # Only show label if bar is visible
                ax.text(x_pos + offset, percentage + max_val * 0.02, 
                        f'{percentage:.1f}%', 
                        ha='center', va='bottom', 
                        fontsize=ANNOTATION_FONTSIZE, 
                        fontweight='bold', zorder=4)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Percentage of Tokens (%)', fontsize=LABEL_FONTSIZE, fontweight='bold')
        
        exit_rate = early_exits / total_tokens if total_tokens > 0 else 0
        
        ax.set_title(f'{difficulty} (Exit rate: {exit_rate:.1%})', fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10, loc='left')
        
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(False)
        if idx == 0 and difficulty in teacher_probs_by_difficulty:
            ax.legend(loc='upper left', fontsize=ANNOTATION_FONTSIZE, framealpha=0.9)
    
    all_x_layers = sorted(list(all_layers)) + [final_layer_pos]
    axes[-1].set_xticks(all_x_layers)
    labels = [str(layer) for layer in all_x_layers[:-1]] + ['Final Layer']
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel('Exit Layer', fontsize=LABEL_FONTSIZE, fontweight='bold')
    
    fig.suptitle('Distribution of Exit Layers by Difficulty', 
                 fontsize=TITLE_FONTSIZE + 2, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate GSM8K with early exit tracking')
    parser.add_argument('--output_plot', type=str, default='gsm8k_exit_distribution.png',
                        help='Output path for the plot')
    parser.add_argument('--output_csv', type=str, default='gsm8k_exit_results.csv',
                        help='Output path for detailed results CSV')
    
    args = parser.parse_args()

    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config_path = "config_deepseek.yaml"
    device = "cuda"
    model_path = "models/early_exit_20251121_kl1_layers_5_big"
    batch_size = 1
    max_samples = 50
    
    tokenizer = get_tokenizer(base_model_name)
    config = configs_from_yaml(config_path, tokenizer.eos_token_id)
    
    base_model = get_model(base_model_name, config['model'], device)
    model = replace_attention_layers(base_model, config['lora'], device)

    #model = load_model_from_wandb(model, model_path = "models/sft_model_2", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early_exit_20250908_layers_5_big:v0')

    model = load_model(model, model_path)

    system_prompt = 'I am going to give you a math word problem. Solve it step by step, showing your reasoning. After your work, provide your final numerical answer.'
    
    set_transformer_early_exit_mode(model, 'free_generate')

    dataset = load_gsm8k_with_difficulty()
    sample_data = sample_balanced_by_difficulty(dataset, max_samples=max_samples)
    
    teacher_probs_by_difficulty = {'Easy': [], 'Medium': [], 'Hard': []}
    teacher_probs_all = [] 
    
    for sample in sample_data:
        difficulty = sample.get('difficulty_category', 'Medium')
        
        with torch.no_grad():
            teacher_probs = calculate_teacher_exit_probs(
                sample, 
                model.exitable_layer_idxs,
                device=device
            )
            avg_probs = teacher_probs.mean(dim=(0, 1)).cpu().numpy()  # [num_layers+1]
            teacher_probs_by_difficulty[difficulty].append(avg_probs)
            teacher_probs_all.append(avg_probs)
    
    for difficulty in ['Easy', 'Medium', 'Hard']:
        if len(teacher_probs_by_difficulty[difficulty]) > 0:
            teacher_probs_by_difficulty[difficulty] = np.mean(
                teacher_probs_by_difficulty[difficulty], axis=0
            )
        else:
            teacher_probs_by_difficulty[difficulty] = np.zeros(model.total_exitable_layers + 1)
    
    teacher_probs_overall = np.mean(teacher_probs_all, axis=0)
    
    difficulty_counts = defaultdict(int)
    for ex in sample_data:
        difficulty_counts[ex.get('difficulty_category', 'Unknown')] += 1
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")

    all_exit_layers = defaultdict(int)
    total_tokens = 0
    results = []
    
    for i, example in enumerate(sample_data):
        prompt = example['full_user_prompt']
        question = prompt 
        difficulty = example['difficulty_category']
        answer = example.get('ground_truth_answer', 'N/A')
        
        with torch.no_grad():
            response, exit_info = generate_text(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                prefiller="",
                tokenizer=tokenizer,
                generation_config=config['generation'],
                device=model.device
            )
            
            response = response[len(prompt):]
            response = response.replace('<｜begin▁of▁sentence｜>', '').replace('｜begin▁of▁sentence｜', '')
            response = response.replace('<｜end▁of▁sentence｜>', '').replace('｜end▁of▁sentence｜', '')
            last_asst = response.rfind("<｜Assistant｜>")
            if last_asst != -1:
                response = response[last_asst + len("<｜Assistant｜>"):].lstrip()
            
            if len(exit_info) >= 2 and hasattr(exit_info[1], 'shape') and model.early_exit_mode == 'free_generate':
                exit_layers = exit_info[1]
                
                if len(exit_layers.shape) > 1:
                    num_tokens = exit_layers.shape[1]
                    exit_layers_flat = exit_layers.flatten()
                else:
                    num_tokens = len(exit_layers)
                    exit_layers_flat = exit_layers
                
                finite_exits = exit_layers_flat[exit_layers_flat != float('inf')]
                unique_layers, counts = torch.unique(finite_exits, return_counts=True)
                
                layer_distribution = {}
                for layer, count in zip(unique_layers, counts):
                    layer_idx = int(layer.item())
                    count_val = count.item()
                    layer_distribution[layer_idx] = count_val
                    all_exit_layers[layer_idx] += count_val
                
                early_exits = len(finite_exits)
                exit_rate = early_exits / num_tokens if num_tokens > 0 else 0
                
            else:
                num_tokens = len(response)
                early_exits = 0
                exit_rate = 0
                layer_distribution = {}
            
            total_tokens += num_tokens
            
            results.append({
                'sample_id': i + 1,
                'difficulty': difficulty,
                'question': question,
                'response': response,
                'num_tokens': num_tokens,
                'early_exits': early_exits,
                'exit_rate': exit_rate,
                'layer_distribution': str(layer_distribution)
            })
            
            print(f"Sample {i+1}/{len(sample_data)} ({difficulty}): "
                  f"{num_tokens} tokens, {early_exits} early exits ({exit_rate:.1%})")
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    
    plot_exit_distribution(results, teacher_probs_by_difficulty, model.exitable_layer_idxs, args.output_plot)
    
    overall_plot_path = args.output_plot.replace('.png', '_overall.png')
    plot_exit_distribution_overall(results, teacher_probs_overall, model.exitable_layer_idxs, overall_plot_path)


if __name__ == "__main__":
    main()