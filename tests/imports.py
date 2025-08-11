import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
# sys.path.append("../../")

sys.path.append("../")
# sys.path.append("..")
from early_exit.patching.method_patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.generate import format_conversation, transform_conversations
from early_exit.util import module_name_is_layer_base
import numpy as np
from early_exit.util import get_model
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
import random
# from early_exit_teacher.visualization import visualize_tokens_by_exit_layer, create_html_visualization
from IPython.display import HTML, display
from early_exit.util import module_name_is_layer_base
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")
import torch.nn.functional as F
import pandas as pd
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, Qwen2Attention

# Model configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading model: {model_name}")
print(f"Device: {device}")

tokenizer = get_tokenizer(model_name)
model_config_path = "/project/project_465001340/fair_stuff/externalization/config_deepseek.yaml"   # args.model_config_path
config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)

model = get_model(model_name, config['model'], device)
model = replace_attention_layers(model, config['lora'], device)
# set_transformer_early_exit_mode(model, 'off')

# Load tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    

config['generation']['max_new_tokens'] = 10

print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")



prompt = "Explain the concept of recursion in programming."
system_prompt = "You are a helpful programming tutor."
prefiller = ""

pre_transformed_conversation = format_conversation(user_prompts = [prompt], system_prompt=system_prompt)
formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]

set_transformer_early_exit_mode(model, 'free_generate')
externalised_response, (externalised_generated_tokens, gathered_early_exit_layer_idxs) =\
    generate_text(model, prompt, system_prompt, prefiller, tokenizer, config['generation'], device)
print(externalised_response)