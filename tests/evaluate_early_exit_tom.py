import torch
import sys
import os
import argparse
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import pandas as pd
from types import SimpleNamespace

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
#from shared_utils.data import CSVPromptDataset
from early_exit.util import get_model, load_model, CSVPromptDataset
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import system_message, solver
from inspect_ai.model import get_model as get_inspect_model, ChatMessageAssistant, ModelOutput

from inspect_ai.scorer import answer as answer_scorer, accuracy, stderr, mean, model_graded_qa, scorer, Score

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
device = "cuda"
model_path = "models/early_exit_20250817_layers_5_kl1_0/step_1500"

dataset_path = "results_and_data/early_exit_sft_dataset/test/validation_w_answers.csv"
prompt_config_path = "results_and_data/early_exit_sft_dataset/test/prompt_config.json"
batch_size=1

dataset = CSVPromptDataset(dataset_path, prompt_config_path)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=False)

new_system_prompt =  """
I am going to give you a story and a question about the story. Read the following story carefully, understand the characters' actions and perspectives, then answer the question regarding object locations, character knowledge, and beliefs. 
Output format (STRICT):
<think>optional reasoning here.</think>
ANSWER: <short answer>

Constraints:
- The <short answer> MUST be all lowercase (no trailing punctuation).
- Nothing after the ANSWER line.
"""

tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, model_path)

set_transformer_early_exit_mode(model, 'free_generate')
#set_transformer_early_exit_mode(model, 'sft_teacher')

answers_df = pd.read_csv(dataset_path, dtype=str).fillna("")
gold_answers = answers_df["answer"].tolist()

samples = []
max_samples = 100

for i, prompt_batch in enumerate(dataloader):
    if len(samples) >= max_samples:
        break

    prompt = prompt_batch.full_user_prompt
    
    with torch.no_grad():
        response, exit_info = generate_text(
            model=model,
            prompt=prompt,
            system_prompt=new_system_prompt,
            prefiller=dataset.prefiller,
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
        else:
            early_exits = 0
            total_tokens = len(response)
            early_exit_rate = 0
            layer_distribution = {}

    gold = gold_answers[i] if i < len(gold_answers) else ""
    
    sample = Sample(
        id=f"early_exit_{i+1}",
        input=str(prompt),
        target=str(gold),
        metadata={
            "system_prompt": new_system_prompt,
            "total_tokens": total_tokens,
            "early_exits": early_exits,
            "early_exit_rate": early_exit_rate,
            "layer_distribution": layer_distribution,
            "response": response
        }
    )
    samples.append(sample)
    
    print(f"Generated {total_tokens} tokens for response {i+1}/{max_samples}, {early_exits} early exits ({early_exit_rate:.1%})")

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


task = Task(
    dataset=MemoryDataset(samples),
    plan=[replay_response()],
    scorer=[
        coherence_scorer(),
        answer_scorer(pattern="line"),  #if ANSWER: exists
        model_graded_qa(                #LLM fallback checker - the main one to use
            partial_credit=False,
            model="openai/gpt-5",
        ),
        early_exit_rate_scorer(),
    ]
)

eval_results = eval(task, model="openai/gpt-5", log_dir='./eval_logs')

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

keys0 = list(log.samples[0].scores.keys())
coh_key = next((k for k in keys0 if "coherence" in k), None)
ans_key = next((k for k in keys0 if "answer" in k), None)
eer_key = next((k for k in keys0 if "early_exit_rate" in k or "early-exit" in k), None)
llm_key = next((k for k in keys0 if "model_graded" in k or "graded" in k or "qa" in k), None)

avg_coherence  = (sum(s.scores[coh_key].as_float() for s in log.samples) / len(log.samples)) if coh_key else 0.0
avg_answer_acc = (
    sum(1.0 if str(s.scores[ans_key].value).upper() in ("C", "CORRECT") else 0.0 for s in log.samples) / len(log.samples)
) if ans_key else 0.0
avg_llm_acc = (
    sum(1.0 if str(s.scores[llm_key].value).upper() in ("C", "CORRECT") else 0.0 for s in log.samples) / len(log.samples)
) if llm_key else 0.0

avg_exit_rate = (sum(s.scores[eer_key].as_float() for s in log.samples) / len(log.samples)) if eer_key else 0.0

print(f"Average Coherence: {avg_coherence:.3f}")
print(f"Answer Scorer Accuracy: {avg_answer_acc:.1%}")
print(f"LLM-Graded Accuracy: {avg_llm_acc:.1%}")
print(f"Average Early Exit Rate: {avg_exit_rate:.2%}")
print(f"Total Samples: {len(log.samples)}")