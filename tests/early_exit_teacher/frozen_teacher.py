import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from early_exit.patching.method_patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.generate import format_conversation, transform_conversations
from early_exit.util import module_name_is_layer_base
import numpy as np
from early_exit.util import get_model
from shared_utils.load import get_tokenizer, configs_from_yaml
import random
from tests.early_exit_teacher.visualization import visualize_tokens_by_exit_layer, create_html_visualization
from IPython.display import HTML, display


# Model configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Loading model: {model_name}")
print(f"Device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model_config_path = "/project/project_465001340/fair_stuff/externalization/config_deepseek.yaml"                     # args.model_config_path
config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
config['generation']['max_new_tokens'] = 100

print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.float16,  # Use half precision for efficiency
    device_map="auto" if device == 'cuda' else None,
    trust_remote_code=True
)

frozen_model = get_model(model_name, config['model'], device)
frozen_model = replace_attention_layers(frozen_model, config['lora'], device)
set_transformer_early_exit_mode(frozen_model, 'sft_student')

from early_exit.util import module_name_is_layer_base
early_exit_layer_idxs = []
for name, module in model.named_modules():
    if module_name_is_layer_base(name):
        layer_idx = int(name.split('.')[-1])
        early_exit_layer_idxs.append(layer_idx)

early_exit_layer_idxs = torch.tensor(early_exit_layer_idxs, dtype = torch.int32)  # Add inf for final layer
print(f"Early exit layer indices: {early_exit_layer_idxs}")
print(f"Total exitable layers: {len(early_exit_layer_idxs)}")  # Subtract 1 for the inf


config['generation']['max_new_tokens'] = 100
# Define the test prompts
test_prompts = [
    "Explain the concept of recursion in programming.",
    "How does gradient descent work in machine learning?",
    "What are the main causes of climate change?"
]

# Define KL factors to test
KL_FACTORS = [1, 4, 16]

# Storage for all results
all_results = {}

# Process each KL factor
for KL_FACTOR in KL_FACTORS:
    print(f"\n{'='*80}")
    print(f"Testing with KL_FACTOR = {KL_FACTOR}")
    print(f"{'='*80}")
    
    all_results[KL_FACTOR] = {}
    
    # Process each prompt
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n\nProcessing Prompt {prompt_idx + 1}: {prompt}")
        print("-"*60)
        
        # Use the same system prompt and prefiller as before
        system_prompt = "You are a helpful assistant."
        prefiller = ""
        
        # Format the prompt
        pre_transformed_conversation = format_conversation(user_prompts=[prompt], system_prompt=system_prompt)
        formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        prompt_length = input_ids.shape[1]
        
        # Generate tokens with early exit tracking
        current_input = input_ids.clone()
        generated_tokens_manual = []
        chosen_exit_layers = []
        kl_divergences = []  # NEW: Store KL divergences
        
        for step in range(config['generation']['max_new_tokens']):
            with torch.no_grad():
                # Forward pass
                outputs = model(current_input, use_cache=True, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                hidden_states = torch.stack(outputs.hidden_states)
                exit_hidden_states = hidden_states[early_exit_layer_idxs, :, -1, :].transpose(0,1)
                exit_predictions = model.lm_head(exit_hidden_states)
                
                # Calculate KL divergence
                final_predictions = torch.softmax(logits, dim=-1)
                teacher_expanded = final_predictions.unsqueeze(1)  
                early_output_probs = torch.softmax(exit_predictions, dim=-1)
                
                eps = 1e-16
                kl_div = (teacher_expanded * ((teacher_expanded + eps) / (early_output_probs + eps)).log()).sum(-1)
                
                # Apply sigmoid transformation with current KL_FACTOR
                sigmoid_kls = torch.sigmoid(KL_FACTOR * kl_div)
                sigmoid_kls = 2.0 * sigmoid_kls - 1.0
                sigmoid_kls = 1.0 - sigmoid_kls
                
                # Determine exit layer
                predictions = final_predictions
                chosen_exit_layer = -1
                for qdx, exit_layer in enumerate(early_exit_layer_idxs):
                    rand_val = random.random()
                    if rand_val < sigmoid_kls[0, qdx]:
                        predictions = early_output_probs[:, qdx]
                        chosen_exit_layer = exit_layer.item()
                        break
                # import ipdb; ipdb.set_trace()
                kl_divergences.append(kl_div[0, qdx].item() if chosen_exit_layer != -1 else None)
                # Sample next token
                if step > 0:
                    chosen_exit_layers_tensor = torch.tensor(chosen_exit_layers, device=device).unsqueeze(0).float()
                    chosen_exit_layers_tensor = torch.where(
                        chosen_exit_layers_tensor == -1,
                        torch.full_like(chosen_exit_layers_tensor, float('inf')),
                        chosen_exit_layers_tensor
                    )
                    output_scores, _ = frozen_model(current_input, prescribed_exit_layer_idxs=chosen_exit_layers_tensor)
                    next_token = torch.argmax(output_scores.logits[:,-1], dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.argmax(predictions, dim=-1).unsqueeze(-1)
                
                # Check for EOS
                if next_token.item() == config['generation']['eos_token_id']:
                    print(f"EOS token encountered at step {step}")
                    break
                
                # Add token to sequence
                current_input = torch.cat([current_input, next_token], dim=1)
                generated_tokens_manual.append(next_token.item())
                chosen_exit_layers.append(int(chosen_exit_layer))
        
        # Decode tokens and store results
        token_strings = [tokenizer.decode([token], skip_special_tokens=True) for token in generated_tokens_manual]
        generated_text = tokenizer.decode(generated_tokens_manual, skip_special_tokens=True)
        
        # Store results
        # all_results[KL_FACTOR][prompt_idx] = (token_strings, chosen_exit_layers, generated_text)
        # chosen_exit_layers = [layer if layer != -1 else 'Final Layer' for layer in chosen_exit_layers]
        all_results[KL_FACTOR][prompt_idx] = (token_strings, chosen_exit_layers, generated_text, kl_divergences)
        # Print summary for this prompt
        print(f"\nGenerated text: {generated_text}")
        print(f"Total tokens: {len(token_strings)}")
        print(f"Exit layer distribution:")
        for layer in early_exit_layer_idxs.tolist() + [-1]:
            count = chosen_exit_layers.count(layer)
            if count > 0:
                percentage = (count / len(chosen_exit_layers) * 100)
                layer_name = f"Layer {layer}" if layer != -1 else "Final Layer"
                print(f"  {layer_name}: {count} tokens ({percentage:.1f}%)")

# Create the HTML visualization
create_html_visualization(
    all_results=all_results,
    early_exit_layer_idxs=early_exit_layer_idxs,
    test_prompts=test_prompts,
    title='Frozen MLP Teacher Early Exit Generation',
    output_path='tests/early_exit_teacher/visualizations/frozen_mlp_teacher_output.html'
)

print("\n\nVisualization complete!")