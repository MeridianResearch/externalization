#!/usr/bin/env python
# coding: utf-8

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append("../")


from shared_utils.data import CSVPromptDataset
from early_exit.util import get_model
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

# import wandb
import pandas as pd
import numpy as np


# LOAD IN EXPERIMENT ARGS
# num_epoch = 1                     # args.num_epoch
num_exit_samples = 1                  # args.num_exit_samples
device = "cuda"                    # args.device
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"                    # args.model_name
model_config_path = "config_deepseek.yaml"                     # args.model_config_path
dataset_path = "../results_and_data/early_exit_sft_dataset/test/data.csv"                  # args.dataset_path
prompt_config_path = "../results_and_data/early_exit_sft_dataset/test/prompt_config.json"                    # args.prompt_config_path
batch_size = 1                    # args.batch_size -- might want to sort out batching, but increasing num_exit_samples might be better + less effort


# LOAD IN THE MODEL AND TOKENIZER
tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
model = get_model(model_name, config['model'], device)


# ENABLE EARLY EXITING
model = replace_attention_layers(model, config['lora'], device)

prompt = "Explain the concept of recursion in programming."
system_prompt = "You are a helpful programming tutor."
prefiller = ""

set_transformer_early_exit_mode(model, 'sft_teacher')
config['generation']['max_new_tokens'] = 10
with torch.no_grad():
    sft_teacher_response, (sft_teacher_generated_tokens, 
                          sft_teacher_final_layer_logprobs, 
                          gathered_early_exit_hidden_states) = generate_text(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        prefiller=prefiller,
        tokenizer=tokenizer,
        generation_config=config['generation'],
        device=device
    )

    early_output_log_probs = model.early_exit_hidden_state_readout(gathered_early_exit_hidden_states)

    early_exit_probs = model.early_exit_target_probs(
       early_output_log_probs=early_output_log_probs,
       teacher_final_layer_log_probs=sft_teacher_final_layer_logprobs
    )





sft_teacher_generated_tokens = sft_teacher_generated_tokens[:, :-1] # Removing the last token to match the log probs and hidden states shape
# ## Testing SFT student
batch, gen_len, elayers = early_exit_probs.shape                                                                                                # [batch, generation length, exitable layers]
full_len = sft_teacher_generated_tokens.shape[1]
repeated_sft_teacher_generated_tokens = sft_teacher_generated_tokens.expand(num_exit_samples * batch, full_len)
# [batch * samples, full length]

with torch.no_grad():
    batch, gen_len, elayers = early_exit_probs.shape 
    full_len = sft_teacher_generated_tokens.shape[1]
    repeated_sft_teacher_generated_tokens = sft_teacher_generated_tokens.expand(num_exit_samples * batch, full_len)   
    sampled_early_exit_layer_idxs_early_with_sample_dim = torch.distributions.Categorical(probs = early_exit_probs).sample((num_exit_samples,))     # [samples, batch, generation length] 
    sampled_early_exit_layer_idxs_early = sampled_early_exit_layer_idxs_early_with_sample_dim.reshape(batch * num_exit_samples, gen_len)            # [batch * samples, generation length]
    sampled_early_exit_layer_idxs = model.exitable_layer_idxs[sampled_early_exit_layer_idxs_early.cpu()]                       
    
    
    set_transformer_early_exit_mode(model, 'sft_student')

    # Create prescribed exit layer idxs filled with torch.inf (always exit on last layer)
    batch_samples, seq_len = repeated_sft_teacher_generated_tokens.shape
    # print("Setting exit layers to inf for sft_student")
    # sampled_early_exit_layer_idxs = torch.full((batch_samples, gen_len), torch.inf, \
    #                                         device=repeated_sft_teacher_generated_tokens.device)
    # print(f"Minimum in prescribed_exit_layer_idxs = {torch.min(sampled_early_exit_layer_idxs)}")
    sft_student_output_scores, collected_exit_logits = model(repeated_sft_teacher_generated_tokens,\
                                                             prescribed_exit_layer_idxs=sampled_early_exit_layer_idxs)
