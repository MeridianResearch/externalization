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
import asyncio
from inspect_ai.model import get_model as get_inspect_model

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode


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


async def evaluate_coherence(prompt: str, response: str) -> dict:

    eval_prompt = f"""
You are evaluating the coherence and quality of an AI assistant's chain of thought and response.
Allow exploratory/meandering reasoning as long as it stays internally consistent.

Original Question: {prompt}

AI Response: {response}

Please evaluate this response on the following criteria:

1. Coherence and logical flow (1-10 scale) Does the reasoning follow a sensible progression, even if unfinished?
2. Completeness of the reasoning (1-10 scale) Does it reach a correct and explicit chain of thought? If partial but on-track and relevant to answer, award mid-range (4–7).
3. Clarity and readability (1-10 scale) Is it easy to follow? Minor meandering is okay if readable.
4. Absence of repetition or errors (1-10 scale) Penalize contradictions, factual mistakes about the prompt, or heavy repetition.

Rate each criterion and provide an overall score from 1-10:
- 1: major breakdown (nonsensical, off-topic)
- 4: noticeable issues but some useful reasoning
- 7: generally solid, with minor flaws or cut-offs
- 10: excellent, complete, and polished

Meta / Wrapper Policy:
- The evaluation input may include wrapper/markup such as: angle-bracket role tags (e.g., <｜User｜>, <｜Assistant｜>) and <think>.
- These wrappers are expected and should not reduce scores for Clarity, Coherence, or No Repetition.

Format your response as:
Coherence: X/10
Completeness: X/10
Clarity: X/10
No Repetition: X/10
Overall: X/40
Brief explanation: [your reasoning]
"""
    
    judge_model = get_inspect_model("openai/gpt-5")
    eval_result = await judge_model.generate(eval_prompt)
    
    eval_text = eval_result.completion
    
    coherence_score = 0
    completeness_score = 0
    clarity_score = 0
    no_repetition_score = 0
    overall_score = 0
    
    for line in eval_text.split('\n'):
        line = line.strip()
        if line.startswith('Coherence:'):
            coherence_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Completeness:'):
            completeness_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Clarity:'):
            clarity_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('No Repetition:'):
            no_repetition_score = int(line.split(':')[1].strip().split('/')[0])
        elif line.startswith('Overall:'):
            overall_score = int(line.split(':')[1].strip().split('/')[0])
    
    explanation = ""
    explanation_start = eval_text.find("Brief explanation:")
    if explanation_start != -1:
        explanation = eval_text[explanation_start + len("Brief explanation:"):].strip()
    else:
        explanation = eval_text
    
    return {
        'coherence': coherence_score,
        'completeness': completeness_score,
        'clarity': clarity_score,
        'no_repetition': no_repetition_score,
        'average': overall_score / 4.0 if overall_score > 0 else (coherence_score + completeness_score + clarity_score + no_repetition_score) / 4.0,
        'explanation': explanation
    }


def clean_response(response: str, prompt: str) -> str:
    response = response[len(prompt):]
    response = response.replace('<｜begin▁of▁sentence｜>', '').replace('｜begin▁of▁sentence｜', '')
    response = response.replace('<｜end▁of▁sentence｜>', '').replace('｜end▁of▁sentence｜', '')
    last_asst = response.rfind("<｜Assistant｜>")
    if last_asst != -1:
        response = response[last_asst + len("<｜Assistant｜>"):].lstrip()
    return response


async def evaluate_model(model_path: str, kl_factor: str, samples: List[Dict], 
                        base_model_name: str, config: Dict, tokenizer, 
                        system_prompt: str, device: str) -> List[Dict]:

    print(f"Evaluating model: {model_path} (KL={kl_factor})")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    base_model = get_model(base_model_name, config['model'], device)
    model = replace_attention_layers(base_model, config['lora'], device)
    model = load_model(model, model_path)
    set_transformer_early_exit_mode(model, 'free_generate')
    #set_transformer_early_exit_mode(model, 'sft_teacher')
    
    if torch.cuda.is_available():
        print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    results = []
    
    for i, example in enumerate(samples):
        prompt = example['full_user_prompt']
        difficulty = example['difficulty_category']
        
        print(f"[KL={kl_factor}] Sample {i+1}/{len(samples)} ({difficulty})")
        
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
        
        cleaned_response = clean_response(response, prompt)
        
        exit_rate_tokens = 0.0
        avg_computation = 0.0
        layer_distribution = {}
        
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
            
            for layer, count in zip(unique_layers, counts):
                layer_idx = int(layer.item())
                count_val = count.item()
                layer_distribution[layer_idx] = count_val
            
            early_exits_count = len(finite_exits)
            exit_rate_tokens = early_exits_count / num_tokens if num_tokens > 0 else 0.0
            
            total_layers = model.config.num_hidden_layers  # e.g., 28
            
            computation_per_token = []
            for layer_idx in exit_layers_flat:
                if layer_idx == float('inf'):
                    computation_per_token.append(1.0)
                else:
                    layers_used = int(layer_idx.item()) + 1  # +1 because layer indices are 0-indexed
                    computation_per_token.append(layers_used / total_layers)
            
            avg_computation = sum(computation_per_token) / len(computation_per_token) if computation_per_token else 1.0
        
        try:
            coherence_result = await evaluate_coherence(prompt, cleaned_response)
            print(f"  Exit Rate (tokens): {exit_rate_tokens:.1%}, Avg Computation: {avg_computation:.2%}")
            print(f"  Coherence: {coherence_result['coherence']}/10, "
                  f"Completeness: {coherence_result['completeness']}/10, "
                  f"Clarity: {coherence_result['clarity']}/10, "
                  f"No Rep: {coherence_result['no_repetition']}/10, "
                  f"Avg: {coherence_result['average']:.2f}")
        except Exception as e:
            print(f"  Error evaluating coherence: {e}")
            coherence_result = {
                'coherence': 0,
                'completeness': 0,
                'clarity': 0,
                'no_repetition': 0,
                'average': 0.0,
                'explanation': f"Error: {str(e)}"
            }
        
        results.append({
            'sample_id': i + 1,
            'difficulty': difficulty,
            'prompt': prompt,
            'response': cleaned_response,
            'exit_rate_tokens': exit_rate_tokens,
            'avg_computation': avg_computation,
            'layer_distribution': str(layer_distribution),
            'coherence': coherence_result['coherence'],
            'completeness': coherence_result['completeness'],
            'clarity': coherence_result['clarity'],
            'no_repetition': coherence_result['no_repetition'],
            'average_score': coherence_result['average'],
            'explanation': coherence_result['explanation']
        })
    
    try:
        if hasattr(model, 'unload'):
            model.unload()
        if hasattr(model, 'delete_adapter'):
            for adapter_name in list(model.peft_config.keys()):
                model.delete_adapter(adapter_name)
    except:
        pass
    
    del model
    del base_model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    import gc
    gc.collect() 
    
    if torch.cuda.is_available():
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    return results

async def main():
    parser = argparse.ArgumentParser(description='Evaluate coherence across different KL factor models')
    parser.add_argument('--output_csv', type=str, default='coherence_comparison_base.csv',
                        help='Output path for the comparison CSV')
    parser.add_argument('--max_samples', type=int, default=50,
                        help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()

    models = {
        'kl0.25': 'models/early_exit_20251121_kl0.25_layers_5_big',
        'kl0.5': 'models/early_exit_20251121_kl0.5_layers_5_big',
        'kl1': 'models/early_exit_20251121_kl1_layers_5_big',
        'kl2': 'models/early_exit_20251121_kl2_layers_5_big',
        'kl4': 'models/early_exit_20251121_kl4_layers_5_big',
    }
    
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    config_path = "config_deepseek.yaml"
    device = "cuda"
    system_prompt = 'I am going to give you a math word problem. Solve it step by step, showing your reasoning. After your work, provide your final numerical answer.'
    
    tokenizer = get_tokenizer(base_model_name)
    config = configs_from_yaml(config_path, tokenizer.eos_token_id)
    
    dataset = load_gsm8k_with_difficulty()
    samples = sample_balanced_by_difficulty(dataset, max_samples=args.max_samples)

    #existing_df = pd.read_csv('coherence_comparison.csv')
    #samples = []
    #for idx, row in existing_df.iterrows():
    #    samples.append({
    #        'full_user_prompt': row['prompt'],
    #        'difficulty_category': row['difficulty']
    #    })
    
    all_results = {}
    for kl_factor, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist, skipping...")
            continue
        
        results = await evaluate_model(
            model_path=model_path,
            kl_factor=kl_factor,
            samples=samples,
            base_model_name=base_model_name,
            config=config,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            device=device
        )
        
        all_results[kl_factor] = results
    
    base_df = pd.DataFrame([
        {
            'sample_id': r['sample_id'],
            'difficulty': r['difficulty'],
            'prompt': r['prompt']
        }
        for r in all_results[list(all_results.keys())[0]]
    ])
    
    for kl_factor, results in all_results.items():
        for i, result in enumerate(results):
            base_df.loc[i, f'{kl_factor}_response'] = result['response']
            base_df.loc[i, f'{kl_factor}_exit_rate_tokens'] = result['exit_rate_tokens']
            base_df.loc[i, f'{kl_factor}_avg_computation'] = result['avg_computation']
            base_df.loc[i, f'{kl_factor}_layer_distribution'] = result['layer_distribution']
            base_df.loc[i, f'{kl_factor}_coherence'] = result['coherence']
            base_df.loc[i, f'{kl_factor}_completeness'] = result['completeness']
            base_df.loc[i, f'{kl_factor}_clarity'] = result['clarity']
            base_df.loc[i, f'{kl_factor}_no_repetition'] = result['no_repetition']
            base_df.loc[i, f'{kl_factor}_average'] = result['average_score']
            base_df.loc[i, f'{kl_factor}_explanation'] = result['explanation']
    
    base_df.to_csv(args.output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {args.output_csv}")
    print(f"{'='*80}\n")

    print("\nSummary Statistics:")
    print("-" * 80)
    for kl_factor in all_results.keys():
        avg_exit_rate = base_df[f'{kl_factor}_exit_rate_tokens'].mean()
        avg_computation = base_df[f'{kl_factor}_avg_computation'].mean()
        avg_coherence = base_df[f'{kl_factor}_coherence'].mean()
        avg_completeness = base_df[f'{kl_factor}_completeness'].mean()
        avg_clarity = base_df[f'{kl_factor}_clarity'].mean()
        avg_no_rep = base_df[f'{kl_factor}_no_repetition'].mean()
        avg_overall = base_df[f'{kl_factor}_average'].mean()
        
        print(f"\n{kl_factor.upper()}:")
        print(f"  Exit Rate (tokens):  {avg_exit_rate:.1%}")
        print(f"  Avg Computation:     {avg_computation:.1%}")
        print(f"  Coherence:           {avg_coherence:.2f}/10")
        print(f"  Completeness:        {avg_completeness:.2f}/10")
        print(f"  Clarity:             {avg_clarity:.2f}/10")
        print(f"  No Repetition:       {avg_no_rep:.2f}/10")
        print(f"  Average Score:       {avg_overall:.2f}/10")


if __name__ == "__main__":
    asyncio.run(main())