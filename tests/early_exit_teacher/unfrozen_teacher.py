#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("../")

from shared_utils.generate import format_conversation, transform_conversations
from early_exit.util import module_name_is_layer_base
import numpy as np
from early_exit.util import get_model
from shared_utils.load import get_tokenizer, configs_from_yaml
import random
from tests.early_exit_teacher.visualization import create_html_visualization

# Set random seed for reproducibility
# random.seed(42)
# torch.manual_seed(42)
# np.random.seed(42)


def generate_with_early_exit(model, tokenizer, prompt, system_prompt, prefiller, 
                           early_exit_layer_idxs, config, kl_factor, device):
    """
    Generate text with early exit for a given KL factor.
    
    Returns:
        token_strings: List of generated token strings
        chosen_exit_layers: List of exit layer indices
        generated_text: The complete generated text
    """
    
    # Format prompt
    pre_transformed_conversation = format_conversation(user_prompts=[prompt], system_prompt=system_prompt)
    formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    current_input = input_ids.clone()
    generated_tokens = []
    chosen_exit_layers = []
    kl_divergences = []  # NEW: Store KL divergences
    
    for step in range(config['generation']['max_new_tokens']):
        with torch.no_grad():
            # Forward pass
            outputs = model(current_input, use_cache=True, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden_states = torch.stack(outputs.hidden_states)
            exit_hidden_states = hidden_states[early_exit_layer_idxs, :, -1, :].transpose(0, 1)
            exit_predictions = model.lm_head(exit_hidden_states)
            
            # Calculate KL divergence
            final_predictions = torch.softmax(logits, dim=-1)
            teacher_expanded = final_predictions.unsqueeze(1)
            early_output_probs = torch.softmax(exit_predictions, dim=-1)
            
            eps = 1e-16
            kl_div = -(teacher_expanded * (early_output_probs + eps).log()).sum(-1)
            
            # Apply KL factor and sigmoid
            sigmoid_kls = torch.sigmoid(kl_factor * kl_div)
            sigmoid_kls = 2.0 * sigmoid_kls - 1.0
            sigmoid_kls = 1.0 - sigmoid_kls
            
            # Select exit layer
            predictions = final_predictions
            chosen_exit_layer = -1
            for qdx, exit_layer in enumerate(early_exit_layer_idxs):
                rand_val = random.random()
                if rand_val < sigmoid_kls[0, qdx]:
                    predictions = early_output_probs[:, qdx]
                    chosen_exit_layer = int(exit_layer.item())
                    break
            chosen_exit_layers.append(chosen_exit_layer)
            kl_divergences.append(kl_div[0, qdx].item() if chosen_exit_layer != -1 else None)
            # Sample next token
            next_token = torch.argmax(predictions, dim=-1).unsqueeze(-1)
            
            # Check for EOS
            if next_token.item() == config['generation']['eos_token_id']:
                break
            
            # Add token to sequence
            current_input = torch.cat([current_input, next_token], dim=1)
            generated_tokens.append(next_token.item())
    
    # Convert tokens to strings
    token_strings = [tokenizer.decode([token], skip_special_tokens=False) for token in generated_tokens]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return token_strings, chosen_exit_layers, generated_text, kl_divergences


# Main execution
if __name__ == "__main__":
    # Model configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == 'cuda' else None,
        trust_remote_code=True
    )
    
    # Extract early exit layer indices
    early_exit_layer_idxs = []
    for name, module in model.named_modules():
        if module_name_is_layer_base(name):
            layer_idx = int(name.split('.')[-1])
            early_exit_layer_idxs.append(layer_idx)
    
    early_exit_layer_idxs = torch.tensor(early_exit_layer_idxs, dtype=torch.int32)
    print(f"Early exit layer indices: {early_exit_layer_idxs}")
    print(f"Total exitable layers: {len(early_exit_layer_idxs)}")
    
    # Load model config
    model_config_path = "config_deepseek.yaml"
    config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
    config['generation']['max_new_tokens'] = 100
    
    # Define test sentences
    test_prompts = [
        "Explain the concept of recursion in programming.",
        # "What is the difference between a list and a tuple in Python?",
        # "How does gradient descent work in machine learning?",
        # "Describe the process of photosynthesis in plants.",
        "What are the main causes of climate change?"
    ]
    
    system_prompt = "You are a helpful assistant that provides clear and concise explanations."
    prefiller = ""
    
    # KL strength values to test
    kl_strengths = [0.25, 0.5, 1, 2, 4, 8]
    
    # Store all results
    all_results = {}
    
    print("\nGenerating text with different KL strengths...")
    print("=" * 60)
    
    for kl_strength in kl_strengths:
        print(f"\nKL Strength: {kl_strength}")
        print("-" * 40)
        
        all_results[kl_strength] = {}
        
        for idx, prompt in enumerate(test_prompts):
            print(f"  Processing prompt {idx + 1}: '{prompt[:50]}...'")
            
            token_strings, exit_layers, generated_text, kl_divergences = generate_with_early_exit(
                model, tokenizer, prompt, system_prompt, prefiller,
                early_exit_layer_idxs, config, kl_strength, device
            )
            
            all_results[kl_strength][idx] = (token_strings, exit_layers, prompt, kl_divergences)
            
            # Print summary statistics
            total_tokens = len(token_strings)
            early_exits = sum(1 for layer in exit_layers if layer != -1)
            early_exit_percentage = (early_exits / total_tokens * 100) if total_tokens > 0 else 0
            
            print(f"    Tokens: {total_tokens}, Early exits: {early_exits} ({early_exit_percentage:.1f}%)")
    
    # Create HTML visualization
    print("\nCreating HTML visualization...")
    create_html_visualization(all_results, early_exit_layer_idxs, test_prompts, 
                            'tests/early_exit_teacher/unfrozen_teacher_output.html')
    
    print("\nVisualization complete!")
    