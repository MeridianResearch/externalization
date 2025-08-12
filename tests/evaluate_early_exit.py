import torch
import sys
import os
import argparse
from typing import List, Dict, Any

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import scorer, accuracy, Score
from inspect_ai.solver import system_message
from inspect_ai.model import get_model as get_inspect_model

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
device = "cuda"
model_path = "models/early_exit_20250811_layers_5_lora32/step_2000"

test_prompts = [
    "Explain the concept of recursion in programming.",
    "Explain what object-oriented programming means using a simple example.",
    "How do neural networks learn from data?",
    "Write a simple Python function to calculate factorial and explain how it works.",
    "Describe the benefits of using version control systems like Git.",
    "What are the key differences between supervised and unsupervised learning?",
    "Describe how HTTP works and what happens when you visit a website."
]

system_prompt = "You are a helpful programming tutor."

tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, model_path)

set_transformer_early_exit_mode(model, 'free_generate')

samples = []

for i, prompt in enumerate(test_prompts):
    
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
        
        if len(exit_info) >= 2 and hasattr(exit_info[1], 'shape'):
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
            total_tokens = 0
            early_exit_rate = 0
            layer_distribution = {}
    
    sample = Sample(
        id=f"early_exit_{i+1}",
        input=prompt,
        target="",
        metadata={
            "system_prompt": system_prompt,
            "total_tokens": total_tokens,
            "early_exits": early_exits,
            "early_exit_rate": early_exit_rate,
            "layer_distribution": layer_distribution,
            "response": response
        }
    )
    samples.append(sample)
    
    print(f"Generated {total_tokens} tokens for response {i+1}/{len(test_prompts)}, {early_exits} early exits ({early_exit_rate:.1%})")

if not samples:
    print("No samples generated successfully!")
    sys.exit(1)

@scorer(metrics=[accuracy()])
def coherence_scorer():
    """Custom scorer for evaluating response coherence"""
    async def score(state, target):
        response = state.metadata.get("response", "")
        prompt = state.input
        early_exit_rate = state.metadata.get("early_exit_rate", 0)
        
        eval_prompt = f"""
You are evaluating the coherence and quality of an AI assistant's response.

Original Question: {prompt}

AI Response: {response}

Please evaluate this response on the following criteria:
1. Coherence and logical flow (1-10 scale)
2. Completeness of the answer (1-10 scale)  
3. Clarity and readability (1-10 scale)
4. Absence of repetition or errors (1-10 scale)

Rate each criterion and provide an overall score from 1-10:
- 1: Poor (major issues)
- 4: Fair (some issues)
- 7: Good (minor issues)
- 10: Excellent (no issues)

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
        
        # Parse overall score
        eval_text = eval_result.completion
        print(f"DEBUG - Raw evaluation text:\n{eval_text}\n" + "="*50)
        overall_score = 0
        
        for line in eval_text.split('\n'):
            if 'Overall:' in line:
                try:
                    score_part = line.split(':')[1].strip()
                    overall_score = int(score_part.split('/')[0])
                    break
                except:
                    pass
        
        # Convert to 0-1 scale
        accuracy_score = overall_score / 40.0 if overall_score > 0 else 0
        
        return Score(
            value=accuracy_score,
            answer=str(overall_score),
            explanation=f"Coherence evaluation (Early exit rate: {early_exit_rate:.1%}): {eval_text}"
        )
    
    return score

task = Task(
    dataset=MemoryDataset(samples),
    plan=[
        system_message("You are evaluating AI responses for coherence and quality.")
    ],
    scorer=coherence_scorer()
)

eval_results = eval(task, model="openai/gpt-5", log_dir='./eval_logs')

log = eval_results[0]

for i, sample in enumerate(log.samples, 1):
    print(f"\nSAMPLE {i}")
    
    exit_rate = sample.metadata['early_exit_rate']
    total_tokens = sample.metadata['total_tokens']
    early_exits = sample.metadata['early_exits']
    layer_dist = sample.metadata['layer_distribution']
    
    model_response = sample.metadata['response']
    clean_response = model_response.replace('<｜begin▁of▁sentence｜>', '').replace('<｜end▁of▁sentence｜>', '')
    print(f"Response: {clean_response}")

    print(f"Exit Rate: {exit_rate:.1%} ({early_exits}/{total_tokens} tokens)")
    print(f"Layer Distribution: {layer_dist}")
    
    coherence_score = sample.scores['coherence_scorer']
    print(f"Score: {coherence_score.value:.2f}/1.0")
    explanation_lines = coherence_score.explanation.split('\n')
    for line in explanation_lines:
        if line.strip():
            print(f"     {line.strip()}")
    print()

avg_coherence = sum(sample.scores['coherence_scorer'].value for sample in log.samples) / len(log.samples)
avg_exit_rate = sum(sample.metadata['early_exit_rate'] for sample in log.samples) / len(log.samples)

print(f"Average Coherence Score: {avg_coherence:.3f}/1.0")
print(f"Average Early Exit Rate: {avg_exit_rate:.1%}")
print(f"Total Samples: {len(log.samples)}")