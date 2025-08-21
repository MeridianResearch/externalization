import torch
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model_from_wandb
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

model_path = "models/trained_model_v0"
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
device = "cuda"


tokenizer = get_tokenizer(base_model)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
# Download the artifact
model = load_model_from_wandb(model, model_path = model_path, 
                              artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0')

print(f"Model loaded w exitable layers: {model.exitable_layer_idxs}")

set_transformer_early_exit_mode(model, 'free_generate')

prompt = "What is 17 x 19"
system_prompt = "You are a helpful math tutor."
prefiller = ""

config['generation']['max_new_tokens'] = 400
with torch.no_grad():
    free_generate_response, exit_info = generate_text(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        prefiller=prefiller,
        tokenizer=tokenizer,
        generation_config=config['generation'],
        device=device
    )
    
    print(f"Free Generate Response: {free_generate_response}")
    # print(f"Exit info: {exit_info}")
        
