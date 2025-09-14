import torch
import sys
import os
import json
import argparse
import re
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd
from types import SimpleNamespace
from datasets import Dataset
from huggingface_hub import hf_hub_download
from collections import defaultdict

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import system_message, solver
from inspect_ai.model import get_model as get_inspect_model, ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import answer as answer_scorer, accuracy, stderr, mean, model_graded_qa, scorer, Score

def load_gsm8k_with_difficulty():
    enriched_file = hf_hub_download(
        repo_id="lizardp1/gsm8k_early_exit",
        filename="validation_with_difficulty.parquet",
        repo_type="dataset"
    )
    
    dataset = Dataset.from_parquet(enriched_file)
    gsm8k_dataset = {'train': dataset}
    
    return gsm8k_dataset

def sample_balanced_by_difficulty(examples, max_samples=100, random_seed=42):
    if max_samples:
        import random
        random.seed(random_seed)
        
        easy_examples = [ex for ex in examples if ex['difficulty_category'] == 'Easy']
        medium_examples = [ex for ex in examples if ex['difficulty_category'] == 'Medium']
        hard_examples = [ex for ex in examples if ex['difficulty_category'] == 'Hard']
        
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

def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if method == "strict":
        # try \boxed{} format first (Qwen's preferred format)
        solutions = re.findall(r"\\boxed\{([+-]?[0-9\\.\\,]+)\}", solution_str)
            
        # fallback for **Final Answer:** format
        if len(solutions) == 0:
            solutions = re.findall(r"\*\*Final Answer:\*\*[\\s\\n]*([+-]?[0-9\\.\\,]+)", solution_str)
            
        if len(solutions) == 0:
            final_answer = None
        else:
            final_answer = solutions[-1].replace(",", "").replace("$", "")
            
    elif method == "flexible":
        answer = re.findall(r"([+-]?[0-9\\.\\,]+)", solution_str) # looks for any number, with/wo commas
        final_answer = None
        if len(answer) == 0:
            # no reward if no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    final_answer = final_answer.replace(",", "").replace("$", "")
                    break
    
    return final_answer

# Configuration
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
device = "cuda"
model_path = "models/rl_model"
batch_size = 1
max_samples = 100

new_system_prompt = 'I am going to give you a math word problem. Solve it step by step, showing your reasoning. After your work, provide your final numerical answer.'

tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
#model = load_model_from_wandb(model, model_path = "models/sft_model_2", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early_exit_20250908_layers_5_big:v0')
model = load_model(model, model_path)

set_transformer_early_exit_mode(model, 'free_generate')

dataset = load_gsm8k_with_difficulty()
gsm8k_data = dataset['train']

sample_data = sample_balanced_by_difficulty(gsm8k_data, max_samples=max_samples)

samples = []
results_by_difficulty = defaultdict(list)

for i, example in enumerate(sample_data):
    if len(samples) >= max_samples:
        break

    question = example['question']
    answer = example['answer']
    difficulty = example['difficulty_category']
    
    ground_truth = str(answer)
    if "#### " in ground_truth:
        answer_match = re.search(r"#### (.+)", ground_truth)
        if answer_match:
            ground_truth = answer_match.group(1).strip().replace(",", "").replace("$", "")
    
    prompt = question
    
    with torch.no_grad():
        response, exit_info = generate_text(
            model=model,
            prompt=prompt,
            system_prompt=new_system_prompt,
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
        
        if len(exit_info) >= 2 and hasattr(exit_info[1], 'shape') and model.early_exit_mode=='free_generate':
            exit_layers = exit_info[1]
            
            if len(exit_layers.shape) > 1:
                total_tokens = exit_layers.shape[1]
                exit_layers_flat = exit_layers.flatten()
            else:
                total_tokens = len(exit_layers)
                exit_layers_flat = exit_layers
            
            #count early exits
            finite_exits = exit_layers_flat[exit_layers_flat != float('inf')]
            early_exits = len(finite_exits)
            early_exit_rate = early_exits / total_tokens if total_tokens > 0 else 0
            
            #layer distribution
            unique_layers, counts = torch.unique(finite_exits, return_counts=True)
            layer_distribution = {}
            if len(unique_layers) > 0:
                for layer, count in zip(unique_layers, counts):
                    layer_distribution[int(layer.item())] = count.item()
        else:
            early_exits = 0
            total_tokens = len(response)
            early_exit_rate = 0
            layer_distribution = {}

    extracted_answer = extract_solution(response, method="strict")
    is_correct = extracted_answer is not None and extracted_answer == ground_truth
    results_by_difficulty[difficulty].append(is_correct)

    sample = Sample(
        id=f"gsm8k_{i+1}",
        input=str(prompt),
        target=str(ground_truth),
        metadata={
            "system_prompt": new_system_prompt,
            "total_tokens": total_tokens,
            "early_exits": early_exits,
            "early_exit_rate": early_exit_rate,
            "layer_distribution": layer_distribution,
            "response": response,
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "original_answer": answer
        }
    )
    samples.append(sample)
    
    print(f"Sample {i+1}/{max_samples} ({difficulty}): {total_tokens} tokens, {early_exits} early exits ({early_exit_rate:.1%}), Correct: {is_correct}")

if not samples:
    print("No samples generated successfully!")
    sys.exit(1)

@scorer(name="coherence_scorer", metrics=[mean(), stderr()])
def coherence_scorer():
    
    async def score(state, target):
        response = state.metadata.get("response", "")
        prompt = state.input
        early_exit_rate = state.metadata.get("early_exit_rate", 0)
        
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
        overall_score = 0
        
        for line in eval_text.split('\n'):
            if 'Overall:' in line:
                try:
                    score_part = line.split(':')[1].strip()
                    overall_score = int(score_part.split('/')[0])
                    break
                except:
                    pass
        
        value = overall_score / 40.0 if overall_score > 0 else 0
        
        return Score(
            value=value,
            answer=str(overall_score),
            explanation=f"Coherence evaluation (Early exit rate: {early_exit_rate:.1%}): {eval_text}"
        )
    
    return score

@scorer(name="gsm8k_accuracy", metrics=[mean(), stderr()])
def gsm8k_accuracy_scorer():
    async def score(state, target):
        is_correct = state.metadata.get("correct", False)
        extracted = state.metadata.get("extracted_answer", "")
        ground_truth = state.metadata.get("ground_truth", "")
        
        return Score(
            value=1.0 if is_correct else 0.0,
            answer="C" if is_correct else "I",
            explanation=f"Extracted: '{extracted}', Ground truth: '{ground_truth}', Correct: {is_correct}"
        )
    
    return score

@solver
def replay_response():
    async def solve(state, generate, *_, **__):
        text = state.metadata.get("response", "")

        state.messages = (state.messages or [])
        state.messages.append(ChatMessageAssistant(content=text))

        state.output = ModelOutput(
            completion=text,
            messages=[ChatMessageAssistant(content=text)],
            tools=[],
            tool_choice=None,
        )

        state.completed = True
        return state
    return solve

@scorer(name="early_exit_rate", metrics=[mean(), stderr()])
def early_exit_rate_scorer():
    async def score(state, target):
        v = float(state.metadata.get("early_exit_rate", 0.0))
        return Score(value=v, explanation=f"early_exit_rate={v:.6f}")
    return score

# Create and run task
task = Task(
    dataset=MemoryDataset(samples),
    plan=[replay_response()],
    scorer=[
        coherence_scorer(),
        gsm8k_accuracy_scorer(),
        early_exit_rate_scorer(),
    ]
)

eval_results = eval(task, model="openai/gpt-5", log_dir='./eval_logs_gsm8k')

log = eval_results[0]

#metrics by difficulty
difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'coherence': [], 'exit_rates': []})

for sample in log.samples:
    difficulty = sample.metadata['difficulty']
    is_correct = sample.metadata['correct']
    coherence = sample.scores['coherence_scorer'].as_float()
    exit_rate = sample.scores['early_exit_rate'].as_float()
    
    difficulty_stats[difficulty]['total'] += 1
    if is_correct:
        difficulty_stats[difficulty]['correct'] += 1
    difficulty_stats[difficulty]['coherence'].append(coherence)
    difficulty_stats[difficulty]['exit_rates'].append(exit_rate)

total_correct = 0
total_samples = 0
all_coherence = []
all_exit_rates = []

for difficulty in sorted(difficulty_stats.keys()):
    stats = difficulty_stats[difficulty]
    accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    avg_coherence = sum(stats['coherence']) / len(stats['coherence']) if stats['coherence'] else 0
    avg_exit_rate = sum(stats['exit_rates']) / len(stats['exit_rates']) if stats['exit_rates'] else 0
    
    print(f"\n{difficulty}:")
    print(f"  Accuracy: {stats['correct']}/{stats['total']} = {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Avg Coherence: {avg_coherence:.3f}")
    print(f"  Avg Early Exit Rate: {avg_exit_rate:.2%}")
    
    total_correct += stats['correct']
    total_samples += stats['total']
    all_coherence.extend(stats['coherence'])
    all_exit_rates.extend(stats['exit_rates'])

# Overall results
overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
overall_coherence = sum(all_coherence) / len(all_coherence) if all_coherence else 0
overall_exit_rate = sum(all_exit_rates) / len(all_exit_rates) if all_exit_rates else 0

print(f"\nOverall:")
print(f"  Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
print(f"  Avg Coherence: {overall_coherence:.3f}")
print(f"  Avg Early Exit Rate: {overall_exit_rate:.2%}")

results_summary = {
    'model_path': model_path,
    'total_samples': total_samples,
    'overall_accuracy': overall_accuracy,
    'overall_coherence': overall_coherence,
    'overall_exit_rate': overall_exit_rate,
    'results_by_difficulty': {}
}

for difficulty, stats in difficulty_stats.items():
    results_summary['results_by_difficulty'][difficulty] = {
        'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
        'total_samples': stats['total'],
        'correct_samples': stats['correct'],
        'avg_coherence': sum(stats['coherence']) / len(stats['coherence']) if stats['coherence'] else 0,
        'avg_exit_rate': sum(stats['exit_rates']) / len(stats['exit_rates']) if stats['exit_rates'] else 0,
    }

with open('gsm8k_eval_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
