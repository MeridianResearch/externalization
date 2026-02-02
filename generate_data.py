import torch
import sys
sys.path.append("../")
sys.path.insert(0, "../..")
from shared_utils.generate import format_conversation, transform_conversations
from early_exit.util import module_name_is_layer_base
import numpy as np
import matplotlib.pyplot as plt
from shared_utils.data import CSVPromptDataset
from shared_utils.load import get_model, get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from shared_utils.load import get_model, get_tokenizer, configs_from_yaml
import random

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append("../")

from shared_utils.load import get_model, get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from early_exit.rl_utils import generate_k_completions_batched
from early_exit.patching import set_transformer_early_exit_mode

import wandb
import pandas as pd
import numpy as np
import string
import html
import matplotlib.colors as mcolors


import torch
import sys
sys.path.append("../")
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset
from typing import Optional
import asyncio
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
from early_exit.util import get_model, load_model_from_wandb, load_model, configs_from_json, save_model
from early_exit.rl_utils import apply_masking, create_attention_mask_from_tokens,  center_rewards_per_prompt, map_layers_to_indices, weighted_sft_step, get_input_prompt_length, evaluate_coherence, compute_sample_labels, load_gsm8k_with_difficulty, compute_accuracy_by_difficulty
from early_exit.util import get_model, load_model_from_wandb, load_model, configs_from_json, save_model, CSVPromptDataset
from early_exit.rl_utils import apply_masking, create_attention_mask_from_tokens, generate_k_completions, center_rewards_per_prompt, map_layers_to_indices, weighted_sft_step, get_input_prompt_length, evaluate_coherence, compute_sample_labels, load_gsm8k_with_difficulty, compute_accuracy_by_difficulty, weighted_sft_loss, compute_entropy_from_logits, create_attention_mask_from_lengths
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import compute_verification_rewards, compute_verification_rewards_text, compute_token_kl_from_logprobs, compute_token_logprobs_reference, compute_token_logprobs_student, compute_avg_exit_layer, extract_solution
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import snapshot_download

from peft import PeftModel

import os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import csv

import csv
import torch
import numpy as np

device = "cpu"
model_name = "Qwen/Qwen3-4B"
config_path = "config_qwen3.yaml"  

tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

# Initialize base student model
student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)

# Load pre-RL model
model_preRL = load_model_from_wandb(
    student, 
    model_path="models/sft_model", 
    artifact_path='vkarthik095-university-of-amsterdam/early-exit/model-checkpoints-sft:v0'
)
set_transformer_early_exit_mode(model_preRL, 'free_generate')

# Re-initialize student for post-RL model
student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)

# Load post-RL model
model_postRL = load_model_from_wandb(
    student, 
    model_path="models/sft_model", 
    artifact_path=(
        "vkarthik095-university-of-amsterdam/"
        "early-exit-RL-test/model-checkpoints-lambda-0_1:v1"
    )
)
set_transformer_early_exit_mode(model_postRL, 'free_generate')

system_prompt = "I am going to give you a story and a question about the story. Read the following story carefully, understand the characters' actions and perspectives, then answer the question regarding object locations, character knowledge, and beliefs.Output format:Answer: <short noun phrase, lowercase, copied verbatim from the story when possible; no trailing punctuation>"

prompts = [
    "Madison entered the press box. Eric entered the press box. Eric retuned the walkie-talkie to a new frequency. James entered the press box. Eric switched the walkie-talkie to a high power mode. Madison permanently connected the walkie-talkie to the stadium's private channel. Does Madison think that Eric believes that the walkie-talkie is in a high power mode? Answer yes or no",
    
    "Cameron entered the conference room. Owen entered the conference room. Owen affixed a visa sticker to the passport. Owen laminated the passport for protection. Cameron received an additional entry permit for the passport. Cameron left the conference room. Owen left the conference room. Jasmine entered the conference room. Jasmine left the conference room. Jasmine entered the conference room. Owen entered the conference room. Jasmine acquired a work permit for the passport. Does Cameron think that Owen believes that the passport has a visa sticker affixed? Answer yes or no.",
    
    "Samantha entered the crew lounge. Ryan entered the crew lounge. Samantha moved the harmonica to the metal toolbox, which is also located in the crew lounge. Ethan entered the crew lounge. Samantha moved the harmonica to the wooden chest, which is also located in the crew lounge. Mason entered the crew lounge. Samantha told out loud that the harmonica is in the wooden chest. Samantha moved the harmonica to the metal toolbox, which is also located in the crew lounge. Mason moved the harmonica to the canvas duffel bag, which is also located in the crew lounge. In which room does Samantha think that Ryan will search for the metal toolbox?"
]

prefiller = ""
config['generation']['max_new_tokens'] = 200

# Prepare CSV file
csv_filename = "early_exit_results.csv"
results = []

NUM_LAYERS = 36  # Qwen3-4B has 36 layers

def calculate_computation_saved(exit_layers):
    """
    Calculate computation saved based on exit layers.
    Exiting at layer L saves (35 - L) / 36 of computation for that token.
    Returns average computation saved across all tokens.
    """
    total_saved = 0.0
    count = 0
    
    for exit_layer in exit_layers:
        if isinstance(exit_layer, (int, float)) and not (isinstance(exit_layer, float) and (np.isnan(exit_layer) or np.isinf(exit_layer))):
            # Computation saved = (layers skipped) / total layers
            # If we exit at layer L, we skip layers L+1 through 35
            layers_skipped = max(0, 35 - exit_layer)
            saved_fraction = layers_skipped / NUM_LAYERS
            total_saved += saved_fraction
            count += 1
    
    return (total_saved / count * 100) if count > 0 else 0.0  # Return as percentage

with torch.no_grad():
    for prompt in prompts:
        try:
            # Generate with pre-RL model
            response_preRL, exit_info_preRL = generate_text(
                model=model_preRL,
                prompt=prompt,
                system_prompt=system_prompt,
                prefiller=prefiller,
                tokenizer=tokenizer,
                generation_config=config['generation'],
                device=device
            )
            
            # Extract pre-RL data
            token_ids_preRL = exit_info_preRL[0].squeeze().tolist()
            exit_layers_preRL = exit_info_preRL[1].squeeze().tolist()
            
            if isinstance(token_ids_preRL, int):
                token_ids_preRL = [token_ids_preRL]
            if isinstance(exit_layers_preRL, (int, float)):
                exit_layers_preRL = [exit_layers_preRL]
            
            # Convert tokens to text (only for generated tokens, matching exit_layers length)
            # Assuming exit_layers corresponds to generated tokens only
            num_generated = len(exit_layers_preRL)
            generated_token_ids_preRL = token_ids_preRL[-num_generated:] if num_generated > 0 else []
            tokens_text_preRL = [tokenizer.decode([tid]) for tid in generated_token_ids_preRL]
            
            # Convert exit layers, keeping inf as string "inf"
            exit_layers_clean_preRL = []
            for x in exit_layers_preRL:
                if x == float('inf'):
                    exit_layers_clean_preRL.append('inf')
                elif isinstance(x, float) and np.isnan(x):
                    exit_layers_clean_preRL.append('nan')
                else:
                    exit_layers_clean_preRL.append(int(x))
            
            # Calculate computation saved for pre-RL
            comp_saved_preRL = calculate_computation_saved(exit_layers_preRL)
            
            # Generate with post-RL model
            response_postRL, exit_info_postRL = generate_text(
                model=model_postRL,
                prompt=prompt,
                system_prompt=system_prompt,
                prefiller=prefiller,
                tokenizer=tokenizer,
                generation_config=config['generation'],
                device=device
            )
            
            # Extract post-RL data
            token_ids_postRL = exit_info_postRL[0].squeeze().tolist()
            exit_layers_postRL = exit_info_postRL[1].squeeze().tolist()
            
            if isinstance(token_ids_postRL, int):
                token_ids_postRL = [token_ids_postRL]
            if isinstance(exit_layers_postRL, (int, float)):
                exit_layers_postRL = [exit_layers_postRL]
            
            # Convert tokens to text (only for generated tokens)
            num_generated = len(exit_layers_postRL)
            generated_token_ids_postRL = token_ids_postRL[-num_generated:] if num_generated > 0 else []
            tokens_text_postRL = [tokenizer.decode([tid]) for tid in generated_token_ids_postRL]
            
            # Convert exit layers
            exit_layers_clean_postRL = []
            for x in exit_layers_postRL:
                if x == float('inf'):
                    exit_layers_clean_postRL.append('inf')
                elif isinstance(x, float) and np.isnan(x):
                    exit_layers_clean_postRL.append('nan')
                else:
                    exit_layers_clean_postRL.append(int(x))
            
            # Calculate computation saved for post-RL
            comp_saved_postRL = calculate_computation_saved(exit_layers_postRL)
            
            print(f"Pre-RL exit layers sample: {exit_layers_clean_preRL[:20]}")
            print(f"Pre-RL computation saved: {comp_saved_preRL:.2f}%")
            print(f"Post-RL exit layers sample: {exit_layers_clean_postRL[:20]}")
            print(f"Post-RL computation saved: {comp_saved_postRL:.2f}%")
            
            # Store result - convert lists to string representation
            result = {
                'system_prompt': system_prompt,
                'prompt': prompt,
                'response_preRL': response_preRL,
                'token_ids_preRL': str(generated_token_ids_preRL),
                'tokens_text_preRL': str(tokens_text_preRL),
                'exit_layers_preRL': str(exit_layers_clean_preRL),
                'computation_saved_preRL': f"{comp_saved_preRL:.2f}%",
                'response_postRL': response_postRL,
                'token_ids_postRL': str(generated_token_ids_postRL),
                'tokens_text_postRL': str(tokens_text_postRL),
                'exit_layers_postRL': str(exit_layers_clean_postRL),
                'computation_saved_postRL': f"{comp_saved_postRL:.2f}%"
            }
            results.append(result)
            
            print(f"Processed prompt: {prompt[:50]}...")
            print(f"Pre-RL Response: {response_preRL[:100]}...")
            print(f"Post-RL Response: {response_postRL[:100]}...")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing prompt: {prompt[:50]}...")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

# Save to CSV
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'system_prompt', 'prompt',
        'response_preRL', 'token_ids_preRL', 'tokens_text_preRL', 'exit_layers_preRL', 'computation_saved_preRL',
        'response_postRL', 'token_ids_postRL', 'tokens_text_postRL', 'exit_layers_postRL', 'computation_saved_postRL'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"\nResults saved to {csv_filename}")