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
from tests.early_exit_teacher.visualization import visualize_tokens_by_exit_layer
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

prompt = "Explain the concept of recursion in programming."
system_prompt = "You are a helpful programming tutor."
prefiller = ""

pre_transformed_conversation = format_conversation(user_prompts = [prompt], system_prompt=system_prompt)
formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]

from early_exit.util import module_name_is_layer_base
early_exit_layer_idxs = []
for name, module in model.named_modules():
    if module_name_is_layer_base(name):
        layer_idx = int(name.split('.')[-1])
        early_exit_layer_idxs.append(layer_idx)

early_exit_layer_idxs = torch.tensor(early_exit_layer_idxs, dtype = torch.int32)  # Add inf for final layer
print(f"Early exit layer indices: {early_exit_layer_idxs}")
print(f"Total exitable layers: {len(early_exit_layer_idxs)}")  # Subtract 1 for the inf

inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids
prompt_length = input_ids.shape[1]

KL_FACTOR = 1
current_input = input_ids.clone()
generated_tokens_manual = []
chosen_exit_layers = []
config['generation']['max_new_tokens'] = 40
for step in range(config['generation']['max_new_tokens']):
    with torch.no_grad():
        # Forward pass
        outputs = model(current_input, use_cache=True, output_hidden_states=True)
        # print(outputs.logits.shape)
        logits = outputs.logits[:, -1, :]  # Get logits for last token
        hidden_states = torch.stack(outputs.hidden_states)
        exit_hidden_states = hidden_states[early_exit_layer_idxs, :, -1, :].transpose(0,1)
        exit_predictions = model.lm_head(exit_hidden_states)

        # 1. Get KL divergence between early exit and final layers
        final_predictions = torch.softmax(logits, dim=-1)
        teacher_expanded = final_predictions.unsqueeze(1)  
        early_output_probs = torch.softmax(exit_predictions, dim=-1)

        # Sum over vocab -> [batch, exitable layers, sequence]
        # print(teacher_expanded.shape, early_output_probs.shape)
        eps = 1e-16
        kl_div = (teacher_expanded * ((teacher_expanded + eps) / (early_output_probs + eps)).log()).sum(-1)
        # kl_div = - (teacher_expanded * (early_output_probs + eps).log()).sum(-1)

        # 2. Scale KL divergencees by KL_FACTOR and pass through sigmoid (0-1)
        sigmoid_kls = torch.sigmoid(KL_FACTOR * kl_div)  # [batch, exitable layers, sequence]
        sigmoid_kls = 2.0 * sigmoid_kls - 1.0
        sigmoid_kls = 1.0 - sigmoid_kls
        predictions = final_predictions
        chosen_exit_layer = -1
        for qdx, exit_layer in enumerate(early_exit_layer_idxs):
            rand_val = random.random()
            if rand_val < sigmoid_kls[0, qdx]:
                predictions = early_output_probs[:, qdx]
                chosen_exit_layer = exit_layer
                break
        # Sample next token
        # import ipdb;  ipdb.set_trace();
        if step > 1:
            chosen_exit_layers_tensor = torch.tensor(chosen_exit_layers[:-1], device=device).unsqueeze(0).float()  # Add batch dimension
            chosen_exit_layers_tensor = torch.where(
                chosen_exit_layers_tensor == -1,
                torch.full_like(chosen_exit_layers_tensor, float('inf')),
                chosen_exit_layers_tensor
            )
            # print(current_input)
            # print([tokenizer.decode(item, skip_special_tokens=True) for item in current_input.squeeze()])
            output_scores, _ = frozen_model(current_input, prescribed_exit_layer_idxs = chosen_exit_layers_tensor) # [batch * samples, full length, vocabulary]
            
            # print(current_input.shape, chosen_exit_layers_tensor.shape)
            next_token_teacher = torch.argmax(predictions, dim=-1).unsqueeze(-1)
            next_token = torch.argmax(output_scores.logits[:,-1], dim=-1).unsqueeze(-1)
            # print(next_token_teacher, tokenizer.decode(next_token_teacher[0], skip_special_tokens=True))
            # print(next_token, tokenizer.decode(next_token[0], skip_special_tokens=True))
        # import ipdb;  ipdb.set_trace();
        # Check for EOS
        else:
            next_token = torch.argmax(predictions, dim=-1).unsqueeze(-1)
        if next_token.item() == config['generation']['eos_token_id']:
            print(f"EOS token encountered at step {step}")
            break
            
        # Add token to sequence
        current_input = torch.cat([current_input, next_token], dim=1)
        generated_tokens_manual.append(next_token.item())
        
        # Decode and print current token
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        # print(f"Step {step}: Token {next_token.item()} -> '{token_text}'")
        chosen_exit_layers.append(int(chosen_exit_layer))

manual_generated_text = tokenizer.decode(generated_tokens_manual, skip_special_tokens=True)

print(f"Generated text: {manual_generated_text}")
print(f"Chosen exit layers: {chosen_exit_layers}")
# print(f"Total tokens generated: {len(generated_tokens_manual)}")

tokens = [tokenizer.decode([token], skip_special_tokens=True) for token in generated_tokens_manual]
layers = chosen_exit_layers
early_exit_layers = early_exit_layer_idxs.tolist()  # Convert tensor to list if needed

# Display the visualization
display(visualize_tokens_by_exit_layer(tokens, layers, early_exit_layers, 
                                     title="Early Exit Token Generation",  save_html="frozen_teacher_output.html"))
