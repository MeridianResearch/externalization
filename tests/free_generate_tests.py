#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
import sys
sys.path.append("../")
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

model_path = "models/early_exit_20250811_layers_5_big"
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config = "config_deepseek.yaml"
device = "cuda"


# In[33]:


tokenizer = get_tokenizer(base_model)
config = configs_from_yaml(config, tokenizer.eos_token_id)

base_model = get_model(base_model, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, model_path)

print(f"Model loaded w exitable layers: {model.exitable_layer_idxs}")


# In[34]:


set_transformer_early_exit_mode(model, 'free_generate')


# In[37]:


prompt = "Explain the concept of recursion in programming."
# prompt = "What are the main causes of climate change?"
system_prompt = "You are a helpful programming tutor."
prefiller = ""


# In[38]:


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

        print(f"Free Generate Response: {free_generate_response}")
        print(f"Exit info: {exit_info}")

    except Exception as e:
        print(f"Free generate mode failed: {e}")


# In[39]:


from early_exit.util import module_name_is_layer_base
early_exit_layer_idxs = []
for name, module in model.named_modules():
    if module_name_is_layer_base(name):
        layer_idx = int(name.split('.')[-1])
        early_exit_layer_idxs.append(layer_idx)

early_exit_layer_idxs = torch.tensor(early_exit_layer_idxs, dtype = torch.int32)  # Add inf for final layer


# In[41]:


# tokens = [tokenizer.decode([token]) for token in exit_info[0][0,21:]]
# layers = [27 if item == torch.inf or item == -1 else int(item) for item in exit_info[1][0]]
# early_exit_layers = early_exit_layer_idxs.tolist()  # Convert tensor to list if needed
# # Display the visualization
# display(visualize_tokens_by_exit_layer(tokens, layers, early_exit_layers, 
#                                      title="Committed Early Exit Token Generation"))


# In[ ]:




