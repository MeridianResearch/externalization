from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm
from early_exit.patching.method_patching import replace_attention_layers
from early_exit.rewards import extract_last_number, extract_solution
from early_exit.rl_types import RLHyperparams
from early_exit.rl_utils import generate_k_completions, load_gsm8k_with_difficulty
from early_exit.util import configs_from_json, get_model, load_model
from shared_utils.load import configs_from_yaml, get_tokenizer
import pandas as pd
import argparse
@torch.no_grad()
def evaluate_last_n_gsm8k_rollouts(model,
                                   mode: str,
                                   tokenizer,
                                   config,
                                   n: int = 20,
                                   k : int = 1, 
                                   device: str = "cuda"):
    assert mode in ['off', 'free_generate'], "mode must be 'off' or 'free_generate'"
    dataset, difficulty_lookup = load_gsm8k_with_difficulty()
    train_dataset = dataset["train"]
    start = len(train_dataset) - n
    rollout_rows = []
    prompt_rows = [] 
    for i in tqdm(range(start, len(train_dataset))):
        example = train_dataset[i]
        prompt = example["question"]
        diff_info = difficulty_lookup.get(prompt, {"difficulty_category": "Unknown"})
        difficulty_category = diff_info["difficulty_category"]
        correct_answer = example["answer"]
        given_answer = extract_last_number(correct_answer)
        completions, exit_info = generate_k_completions(model, [prompt], k=k, 
                                                        tokenizer=tokenizer, config=config, device=device, 
                                                        system_prompt = RLHyperparams().system_prompt, mode = mode)  # TODO
        rewards = []
        generated_answers = []
        for r in range(k):
            generated_text = completions["texts"][r]
            generated_answer = extract_last_number(generated_text)
            reward = int(generated_answer == given_answer)
            rollout_rows.append({
                "example": i,
                "rollout": r, 
                "samples/prompt_text": prompt,
                "samples/completion_text": generated_text,
                "samples/generated_answer": generated_answer,
                "samples/correct_answer": given_answer,
                "samples/verify_reward": reward,                  # 1 if exact match, else 0
                "samples/difficulty_category": difficulty_category,
                # "samples/kl_estimate": None,                      # not computed in this lightweight eval
                # "samples/avg_exit_layer": float(avg_exit_layer[r].item()),
                # "samples/gen_len": gen_len,
                "samples/contains_eos": tokenizer.eos_token in generated_text,
            })
            
            
            
        #     rewards.append(reward)
        #     generated_answers.append(generated_answer)
            
        # prompt_rows.append({
        #     "example": i,
        #     "samples/prompt_text": prompt,
        #     "samples/correct_answer": given_answer,
        #     "samples/generated_answer": generated_answers,
        #     "samples/difficulty_category": difficulty_category,
        #     "samples/verify_reward": np.mean(rewards),                  # 1 if exact match, else 0
        # })
    # return rollout_rows, prompt_rows
    return rollout_rows

def print_evaluation_summary(
    rollout_rows: List[Dict], 
    mode: str
) -> Dict:
    """
    Print and return a summary of evaluation results.
    
    Args:
        rollout_rows: List of rollout-level evaluation results
        prompt_rows: List of prompt-level evaluation results
        mode: Evaluation mode used
        
    Returns:
        Dictionary containing summary statistics
    """
    # Calculate overall statistics
    all_rewards = [row["samples/verify_reward"] for row in rollout_rows]
    overall_accuracy = np.mean(all_rewards) if all_rewards else 0
    
    # Calculate per-difficulty accuracy
    difficulty_stats = {}    
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY (Mode: {mode})")
    print("="*60)
    print(f"Total rollouts: {len(rollout_rows)}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    if difficulty_stats:
        print("\nAccuracy by difficulty:")
        for diff, rewards in sorted(difficulty_stats.items()):
            print(f"  {diff}: {np.mean(rewards):.2%} (n={len(rewards)})")
    
    # Check for EOS token presence
    eos_count = sum(1 for row in rollout_rows if row["samples/contains_eos"])
    print(f"\nCompletions with EOS token: {eos_count}/{len(rollout_rows)} ({eos_count/len(rollout_rows):.2%})")
    
    # Average generation length
    gen_lens = [row.get("samples/gen_len", 0) for row in rollout_rows if "samples/gen_len" in row]
    if gen_lens:
        print(f"Average generation length: {np.mean(gen_lens):.1f} tokens")
    
    print("="*60 + "\n")
    
    return {
        "mode": mode,
        "overall_accuracy": overall_accuracy,
        "difficulty_stats": {k: float(np.mean(v)) for k, v in difficulty_stats.items()},
        "eos_ratio": eos_count / len(rollout_rows) if rollout_rows else 0,
        "avg_gen_length": np.mean(gen_lens) if gen_lens else 0,
        "total_rollouts": len(rollout_rows)
    }



def safe_filename(path: str) -> str:
    """Convert a path to a safe filename by replacing slashes with underscores."""
    return Path(path).name.replace('/', '_').replace('\\', '_')

@torch.no_grad()
def load_model_tokenizer_config(model_path : str,
                            config_path: str,  
                            base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                            device: str = "cuda",
                            max_new_tokens = 1000):
    
        # --- Models (schema) ---
    tokenizer = get_tokenizer(base_model_name)
    # config = configs_from_yaml(config_path, tokenizer.eos_token_id)

    config = configs_from_yaml(config_path, tokenizer.eos_token_id)
    adapter_config = configs_from_json(model_path + "/early_exiter/adapter_config.json")
    config['lora']['r'] = adapter_config.get('r', config['lora']['r'])
    config['lora']['lora_alpha'] = adapter_config.get('lora_alpha', config['lora']['lora_alpha'])
    config['generation']['max_new_tokens'] = max_new_tokens

    student = get_model(base_model_name, config['model'], device)
    student = replace_attention_layers(student, config['lora'], device)
    # TODO: Change artifact path to sft trained gsm-8k model
    # student = load_model_from_wandb(student, model_path = "models/trained_model_v0", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0')
    student = load_model(student, model_path)
    return student, tokenizer, config
   
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
parser.add_argument("--mode", type=str, choices=["off", "free_generate"], required=True, help="Evaluation mode: 'off' or 'free_generate'")
parser.add_argument("--n", type=int, default=100, help="Number of examples to evaluate (default: 100)")
args = parser.parse_args()
model_path = args.model_path
mode = args.mode
n = args.n
k = 1

print(f"Hyperparameters:")
print(f"  Model path: {model_path}")
print(f"  Mode: {mode}")
print(f"  N examples: {n}")
print(f"  K rollouts: {k}")
print()

config_mode = "greedy"
config_path = f"config_{config_mode}.yaml"

print(f"Hyperparameters:")
print(f"  Model path: {model_path}")
print(f"  Mode: {mode}")
print(f"  N examples: {n}")
print(f"  K rollouts: {k}")
print(f"  Config path: {config_path}")
print()
 
student, tokenizer, config = load_model_tokenizer_config(
    model_path=model_path,
    config_path=config_path,
    base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device="cuda",
    max_new_tokens=1000,
)

rollout_rows = evaluate_last_n_gsm8k_rollouts(
    model=student,
    mode = mode,
    tokenizer=tokenizer,
    config=config,
    n=n,
    k=k,
    device="cuda",
)
print_evaluation_summary(rollout_rows, mode)
df = pd.DataFrame(rollout_rows)
df.to_csv(f"results_and_data/eval/rollout_results_{safe_filename(model_path)}_{mode}_{n}_{k}_{config_mode}.csv", index=False)