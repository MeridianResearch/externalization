import torch
import sys
import os
import argparse
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd
from types import SimpleNamespace
import re

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
#from shared_utils.data import CSVPromptDataset
from early_exit.util import get_model, load_model, CSVPromptDataset, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import system_message, solver
from inspect_ai.model import get_model as get_inspect_model, ChatMessageAssistant, ModelOutput

from inspect_ai.scorer import answer as answer_scorer, accuracy, stderr, mean, model_graded_qa, scorer, Score

base_model_name = "Qwen/Qwen3-4B" #"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
device = "cuda"
model_path = "models/rl_20251212_tom_rlmodel_batch4_k2_lambda1.0/step_200"

dataset_path = "results_and_data/early_exit_sft_dataset/test/tom_eval.csv"
prompt_config_path = "results_and_data/early_exit_sft_dataset/test/prompt_config_tom.json"
batch_size=1

dataset = CSVPromptDataset(dataset_path, prompt_config_path)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=False)

tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, model_path)
#model = load_model_from_wandb(model, model_path = "models/sft_model", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/my-model:v0')
#model = load_model_from_wandb(model, model_path = "models/rl_model", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit-RL-test/model-checkpoints-lambda-0:v0')
#model = load_model_from_wandb(model, model_path = "models/rl_model_1_0", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit-RL-test/model-checkpoints-lambda-0:v1')
#model = load_model_from_wandb(model, model_path = "models/rl_model_0_5", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit-RL-test/model-checkpoints-lambda-0_5:v0')
#model = load_model_from_wandb(model, model_path = "models/rl_model_0_2", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit-RL-test/model-checkpoints-lambda-0_2:v0')
#model = load_model_from_wandb(model, model_path = "models/rl_model_0_1", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit-RL-test/model-checkpoints-lambda-0_1:v0')

total_layers = 36 #hardcoded as we know in this case

set_transformer_early_exit_mode(model, 'free_generate')
#set_transformer_early_exit_mode(model, 'sft_teacher')

answers_df = pd.read_csv(dataset_path, dtype=str).fillna("")
gold_answers = answers_df["answer"].tolist()

samples = []
max_samples = 60

def extract_solution_text(solution_str: str, method: str = "strict"):
    assert method in ["strict", "flexible"]

    if not isinstance(solution_str, str) or not solution_str.strip():
        return None

    s = solution_str

    s = s.replace("<｜begin▁of▁sentence｜>", "").replace("<｜end▁of▁sentence｜>", "")
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"<\|.*?\|>", "", s, flags=re.S)  # remove <|...|> tokens

    if method == "strict":
        parts = re.split(r"<\|\s*Assistant\s*\|>", s, flags=re.I)
        if len(parts) >= 2:
            s = parts[-1]

        anchor = 0
        rs = list(re.finditer(r"^\s*reasoning\s*:\s*", s, flags=re.I | re.M))
        if rs:
            anchor = max(anchor, rs[-1].end())
        s_tail = s[anchor:] if anchor > 0 else s

        matches = list(re.finditer(
            r"^\s*(?:\*\*\s*)?answer(?:\s*\*\*)?\s*[:：]\s*(.+?)\s*$",
            s_tail, flags=re.I | re.M
        ))
        if not matches:
            return None

        ans = matches[-1].group(1).strip()
        ans = re.sub(r"^\*\*|\*\*$", "", ans).strip()
        ans = ans.replace("<｜end▁of▁sentence｜>", "").strip()
        return ans or None

    else:  # method == "flexible"
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        return (lines[-1] if lines else None)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

for i, prompt_batch in enumerate(dataloader):
    if len(samples) >= max_samples:
        break

    prompt = prompt_batch.full_user_prompt
    
    with torch.no_grad():
        response, exit_info = generate_text(
            model=model,
            prompt=prompt,
            system_prompt=dataset.system_prompt,
            prefiller=dataset.prefiller,
            tokenizer=tokenizer,
            generation_config=config['generation'],
            device=model.device
        )
        response = response[len(prompt):]

        completion_tokens = tokenizer.encode(response, add_special_tokens=False)
        completion_length = len(completion_tokens)
        
        response = response.replace('<｜begin▁of▁sentence｜>', '').replace('｜begin▁of▁sentence｜', '')
        response = response.replace('<｜end▁of▁sentence｜>', '').replace('｜end▁of▁sentence｜', '')
        last_asst = response.rfind("<｜Assistant｜>")
        if last_asst != -1:
            response = response[last_asst + len("<｜Assistant｜>"):].lstrip()
        #print(prompt)
        #print(response)
        #print(exit_info)
        
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

            #average computation per token
            computation_per_token = []
            for layer_idx in exit_layers_flat:
                if layer_idx == float('inf'):
                    computation_per_token.append(1.0)
                else:
                    layers_used = int(layer_idx.item()) + 1  # +1 because layer indices are 0-indexed
                    computation_per_token.append(layers_used / total_layers)
            
            avg_computation = sum(computation_per_token) / len(computation_per_token) if computation_per_token else 1.0
            
        else:
            early_exits = 0
            total_tokens = len(response)
            early_exit_rate = 0
            layer_distribution = {}
            avg_computation = 1.0

    extracted_answer = _norm(extract_solution_text(response, method="strict"))
    ground_truth = _norm(gold_answers[i]) if i < len(gold_answers) else ""
    is_correct = extracted_answer is not None and extracted_answer == ground_truth
    
    sample = Sample(
        id=f"early_exit_{i+1}",
        input=str(prompt),
        target=str(ground_truth),
        metadata={
            "system_prompt": dataset.system_prompt,
            "total_tokens": total_tokens,
            "early_exits": early_exits,
            "early_exit_rate": early_exit_rate,
            "layer_distribution": layer_distribution,
            "avg_computation": avg_computation,
            "completion_length": completion_length,
            "response": response,
            "ground_truth": ground_truth,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "response": response
        }
    )
    samples.append(sample)
    
    print(f"Generated {total_tokens} tokens for response {i+1}/{max_samples}, {early_exits} early exits ({early_exit_rate:.1%}), avg_comp: {avg_computation:.3f}, completion_len: {completion_length}")

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

@scorer(name="tom_accuracy", metrics=[mean(), stderr()])
def tom_accuracy_scorer():
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

@scorer(name="avg_computation", metrics=[mean(), stderr()])
def avg_computation_scorer():
    async def score(state, target):
        v = float(state.metadata.get("avg_computation", 1.0))
        return Score(value=v, explanation=f"avg_computation={v:.6f}")
    return score

@scorer(name="completion_length", metrics=[mean(), stderr()])
def completion_length_scorer():
    async def score(state, target):
        v = float(state.metadata.get("completion_length", 0))
        return Score(value=v, explanation=f"completion_length={v:.0f}")
    return score

# Create and run task
task = Task(
    dataset=MemoryDataset(samples),
    plan=[replay_response()],
    scorer=[
        coherence_scorer(),
        tom_accuracy_scorer(),
        early_exit_rate_scorer(),
        avg_computation_scorer(),
        completion_length_scorer(),
    ]
)

eval_results = eval(task, model="openai/gpt-5", log_dir='./eval_logs_tom')

log = eval_results[0]

# for i, sample in enumerate(log.samples, 1):
#     print(f"\nSAMPLE {i}")
    
#     exit_rate = sample.metadata['early_exit_rate']
#     total_tokens = sample.metadata['total_tokens']
#     early_exits = sample.metadata['early_exits']
#     layer_dist = sample.metadata['layer_distribution']
    
#     model_response = sample.metadata['response']
#     prompt = sample.input
#     clean_response = model_response.replace('<｜begin▁of▁sentence｜>', '').replace('<｜end▁of▁sentence｜>', '')
#     print(f"Prompt: {prompt}")
#     print(f"Response: {model_response}")

#     print(f"Exit Rate: {exit_rate:.1%} ({early_exits}/{total_tokens} tokens)")
#     print(f"Layer Distribution: {layer_dist}")
    
#     coherence_score = sample.scores['coherence_scorer']
#     print(f"Score: {coherence_score.value:.2f}/1.0")
#     explanation_lines = coherence_score.explanation.split('\n')
#     for line in explanation_lines:
#         if line.strip():
#             print(f"     {line.strip()}")
#     print()

# ===== Overall results only (no difficulty categories) =====
total_correct = 0
total_samples = 0
all_coherence = []
all_exit_rates = []
all_avg_computation = []
all_total_tokens = []

sample_results = []

for sample in log.samples:
    is_correct = bool(sample.metadata.get('correct', False))
    coherence = sample.scores['coherence_scorer'].as_float()
    exit_rate = sample.scores['early_exit_rate'].as_float()
    avg_comp = sample.scores['avg_computation'].as_float()
    total_toks = sample.scores['completion_length'].as_float()

    total_samples += 1
    total_correct += int(is_correct)
    all_coherence.append(coherence)
    all_exit_rates.append(exit_rate)
    all_avg_computation.append(avg_comp)
    all_total_tokens.append(total_toks)

    sample_results.append({
        'sample_id': sample.id,
        'correct': is_correct,
        'coherence': coherence,
        'exit_rate': exit_rate,
        'avg_computation': avg_comp,
        'total_tokens': total_toks,
        'extracted_answer': sample.metadata.get('extracted_answer', ''),
        'ground_truth': sample.metadata.get('ground_truth', ''),
        'response': sample.metadata.get('response', ''),
    })

overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
overall_coherence = (sum(all_coherence) / len(all_coherence)) if all_coherence else 0.0
overall_exit_rate = (sum(all_exit_rates) / len(all_exit_rates)) if all_exit_rates else 0.0
overall_avg_computation = (sum(all_avg_computation) / len(all_avg_computation)) if all_avg_computation else 0.0
overall_avg_tokens = (sum(all_total_tokens) / len(all_total_tokens)) if all_total_tokens else 0.0
overall_tot_computation = overall_avg_tokens * overall_avg_computation

print("\nOverall:")
print(f"  Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
print(f"  Avg Coherence: {overall_coherence:.3f}")
print(f"  Avg Early Exit Rate: {overall_exit_rate:.2%}")
print(f"  Avg Computation: {overall_avg_computation:.3f}")
print(f"  Avg Total Tokens: {overall_avg_tokens:.1f}")

results_summary = {
    'model_path': model_path,
    'samples': total_samples,
    'coherence': overall_coherence,
    'exit_rate': overall_exit_rate,
    'avg_compute': overall_avg_computation,
    'avg_tokens': overall_avg_tokens,
    'accuracy': overall_accuracy,
    'tot_compute': overall_tot_computation,
}

samples_df = pd.DataFrame(sample_results)
samples_csv_path = './eval_logs_tom/sample_results_4b.csv'
samples_df.to_csv(samples_csv_path, index=False)
print(f"\nPer-sample results saved to: {samples_csv_path}")

# Save summary results to CSV
summary_df = pd.DataFrame([results_summary])
summary_csv_path = './eval_logs_tom/summary_results_4b.csv'
import os
file_exists = os.path.exists(summary_csv_path)

summary_df.to_csv(summary_csv_path, mode='a', header=not file_exists, index=False)
print(f"Summary results {'appended to' if file_exists else 'saved to'}: {summary_csv_path}")