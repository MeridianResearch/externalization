import torch
import sys
sys.path.append("../")
import sys
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


device = "cpu"
model_name ="Qwen/Qwen3-4B"#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_qwen3.yaml"  




tokenizer = get_tokenizer(model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

student = get_model(model_name, config['model'], device)
student = replace_attention_layers(student, config['lora'], device)


model = load_model_from_wandb(student, model_path = "models/sft_model", artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/model-checkpoints-sft:v0')

set_transformer_early_exit_mode(model, 'free_generate')

system_prompt = "I am going to give you a story and a question about the story. Read the following story carefully, understand the characters' actions and perspectives, then answer the question regarding object locations, character knowledge, and beliefs.Output format:Answer: <short noun phrase, lowercase, copied verbatim from the story when possible; no trailing punctuation>"

prompt = " Justin entered the planning room. Justin moved the pocket-sized compass to the wooden desk drawer, which is also located in the planning room. While this action was happening, Cameron witnessed this action in secret (and only this action). Cameron entered the planning room. Cameron filled the pocket-sized compass's small flap compartment with an emergency contact list. Justin moved the pocket-sized compass to the metal lunchbox, which is also located in the planning room. Olivia entered the planning room. In which container does Justin think that Cameron will search for the pocket-sized compass?"
prefiller = ""

config['generation']['max_new_tokens'] = 100
with torch.no_grad():
    try:
        free_generate_response, exit_info = generate_text(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            prefiller=prefiller,
            tokenizer=tokenizer,
            generation_config=config['generation'],
            device=device
        )
        
        print(f"Free Generate Response: {free_generate_response,}")
        print(f"Exit info: {exit_info}")
        
    except Exception as e:
        print(f"Free generate mode failed: {e}")


import html
import matplotlib.colors as mcolors
from IPython.display import HTML

def safe_decode_tokens(tokenizer, tokens):
    try:
        # Ensure tokens are on CPU and converted to a list
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Decoding failed: {e}")
        return ""
import html
import torch
import matplotlib.colors as mcolors
from IPython.display import HTML, display

# --- 1. VISUALIZATION FUNCTION ---
def visualize_tokens_by_exit_layer(token_strings, exit_layers, early_exit_layer_idxs=None, 
                                  title="Token Early Exit Visualization", prompt="", 
                                  save_html=None, limit=None):
    """
    Visualize tokens colored by their early exit layers.
    Expects token_strings to be a LIST of strings, not a single string.
    """
    
    # Slice inputs if limit is set
    if limit is not None:
        token_strings = token_strings[:limit]
        exit_layers = exit_layers[:limit]

    # Get all unique layers to determine the range
    unique_layers = sorted(set(exit_layers))
    if early_exit_layer_idxs is not None:
        # Ensure we cover the full range of possible exit layers in the legend
        all_layers = list(early_exit_layer_idxs) + [max(exit_layers) if exit_layers else 36]
        unique_layers = sorted(set(all_layers))
    
    # Custom Color Setup
    custom_hex_colors = [
        '#6E4C4B',  # Early (Dark Brown/Red)
        '#975654',
        '#D6B886',
        '#EBE3D9',
        '#FAFAFA'   # Late (Almost White)
    ]
    
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_early_exit", custom_hex_colors)
    
    if not unique_layers:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    else:
        norm = mcolors.Normalize(vmin=min(unique_layers), vmax=max(unique_layers))
    
    layer_colors = {}
    for layer in unique_layers:
        rgba = cmap(norm(layer))
        layer_colors[layer] = mcolors.to_hex(rgba)

    # Build HTML
    html_content = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px; padding: 20px; 
                background-color: #f9f9f9; border-radius: 10px;">
        <h3 style="text-align: center; color: #333; margin-bottom: 20px;">{title}</h3>
    """
    
    if prompt:
        html_content += f"""
        <div style="margin: 15px 0; padding: 12px; background-color: #fff3cd; 
                    border-left: 4px solid #D6B886; border-radius: 5px;">
            <strong style="color: #000;">Prompt:</strong> 
            <span style="color: #000;">{html.escape(prompt)}</span>
        </div>
        """
    
    # Legend
    html_content += """
        <div style="display: flex; justify-content: center; gap: 15px; 
                    margin: 20px 0; padding: 15px; background-color: #fff; 
                    border-radius: 5px; flex-wrap: wrap; border: 1px solid #ddd;">
    """
    
    for layer in unique_layers:
        # Only show legend items that actually appear or are major milestones
        if layer in exit_layers or (early_exit_layer_idxs and layer in early_exit_layer_idxs):
            color = layer_colors.get(layer, "#FFFFFF")
            html_content += f"""
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 25px; height: 15px; background-color: {color}; 
                                border: 1px solid #999; border-radius: 3px;"></div>
                    <span style="font-size: 14px; color: #000; font-weight: 500;">Layer {layer}</span>
                </div>
            """
    
    html_content += """
        </div>
        <div style="line-height: 2.5; word-wrap: break-word; padding: 15px; 
                    background-color: #fff; border-radius: 5px; border: 1px solid #ddd;">
    """
    
    # Tokens Loop
    # This is where the fix enables proper zipping
    for token, exit_layer in zip(token_strings, exit_layers):
        color = layer_colors.get(exit_layer, "#FFFFFF")
        
        # Escape HTML characters in the token string
        token_display = html.escape(token, quote=False)
        
        # Visualize whitespace slightly
        token_display = token_display.replace('\n', '<span style="opacity:0.3">\\n</span>')
        token_display = token_display.replace('\t', '<span style="opacity:0.3">\\t</span>')
        
        # Determine text color based on background brightness
        hex_c = color.lstrip('#')
        r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "white" if brightness < 150 else "black"
        
        html_content += f"""<span style="display: inline-block; padding: 4px 8px; margin: 2px; 
                                      border-radius: 4px; border: 1px solid #ccc; 
                                      font-family: monospace; font-size: 13px; 
                                      background-color: {color}; color: {text_color}; 
                                      font-weight: bold; max-width: 200px; 
                                      overflow-wrap: break-word; vertical-align: middle;" 
                                      title="Layer {exit_layer}">{token_display}</span>"""
    
    html_content += "</div>"
        
    layer_counts = {l: exit_layers.count(l) for l in unique_layers if exit_layers.count(l) > 0}
    stats_items = [f"L{l}: {c}" for l, c in layer_counts.items()]
    stats_text = " &nbsp;|&nbsp; ".join(stats_items)
    
    html_content += f"""
        <div style="margin-top: 15px; padding: 10px; background-color: #f0f0f0; 
                    border-radius: 5px; font-family: monospace; font-size: 13px; color: #333;">
            <strong>Showing first {len(token_strings)} tokens</strong> &nbsp;|&nbsp; {stats_text}
        </div>
    </div>
    """
    
    if save_html:
        full_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_content}</body></html>"
        with open(save_html, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"HTML visualization saved to: {save_html}")
        return html_content 
    else:
        return HTML(html_content)

# --- 2. DATA PROCESSING (THE FIX) ---

# Calculate generation length based on the exit info tensor shape
gen_len = exit_info[1][0].shape[-1]

# Get the raw token IDs
raw_token_ids = exit_info[0][0, -gen_len:]

# Ensure it's a list or numpy array, not a tensor on GPU
if hasattr(raw_token_ids, "tolist"):
    raw_token_ids = raw_token_ids.tolist()

tokens = [tokenizer.decode([tid]) for tid in raw_token_ids]

# Process layers (handling inf and -1 as Layer 36)
layers = [36 if item == torch.inf or item == -1 else int(item) for item in exit_info[1][0]]

# Print debug info to confirm shape
print(f"Number of tokens: {len(tokens)}")
print(f"Number of layers: {len(layers)}")

# --- 3. GENERATE VISUALIZATION ---

html_obj = visualize_tokens_by_exit_layer(
    tokens,  # Now passing a list of strings
    layers, 
    [int(item) for item in model.exitable_layer_idxs[:-1]], 
    title="Proof of Concept: Early Exit Mechanism Successfully Engages (after RL)",
    prompt=prompt
)

# Extract raw HTML string regardless of whether return was HTML object or string
html_str = html_obj.data if isinstance(html_obj, HTML) else html_obj

# Try to display (only works in Notebooks)
try:
    display(HTML(html_str))
except:
    pass

# --- 4. SAVE TO IMAGE ---

try:
    from html2image import Html2Image
    hti = Html2Image(output_path='./')
    
    full_html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 20px; background-color: white; }}
        </style>
    </head>
    <body>
        {html_str}
    </body>
    </html>
    """
    
    output_file = 'token_exit_visualization_committed.png'
    hti.screenshot(html_str=full_html_doc, save_as=output_file, size=(1400, 1000))
    print(f"✅ Visualization saved as: {output_file}")
    
except ImportError:
    print("⚠️ html2image not installed. Install with: pip install html2image")
except Exception as e:
    print(f"⚠️ Could not save PNG: {e}")
    print("   Saving as HTML instead...")
    html_file = 'token_exit_visualization_committed.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body>{html_str}</body></html>")
    print(f"✅ Saved as HTML: {html_file}")