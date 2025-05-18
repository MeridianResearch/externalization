import torch

from shared_util.load import get_model, get_tokenizer, configs_from_yaml
from shared_util.generate import generate_text
from shared_util.prompts import BayesianProblemRewardHackingPromptFarm


from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode


################################################################################
########################### GENERATE EXAMPLE DATA ##############################
################################################################################
prompt_farm = BayesianProblemRewardHackingPromptFarm()
batch = prompt_farm.generate_batch(1)
prompt = batch['prompts']
system_prompt = batch['system_prompts']
prefiller = batch['prefillers']


################################################################################
################# LOAD IN THE MODEL AND TOKENIZER ##############################
################################################################################
device = 'cuda'
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "GenPRM/GenPRM-7B"

tokenizer = get_tokenizer(model_name)
default_config_path = "config_deepseek.yaml"
config = configs_from_yaml(default_config_path, tokenizer.eos_token_id)
model = get_model(model_name, config['model'], device)


################################################################################
########################## ENABLE EARLY EXITING ################################
################################################################################
model = replace_attention_layers(model, config['lora']).to(device)


################################################################################
################# GET THE ORIGINAL RESPONSE WITHOUT EARLY EXIT #################
################################################################################
original_response, *_ = generate_text(model, prompt, system_prompt, prefiller, tokenizer, config['generation'], device)
print(original_response)


################################################################################
################## GET THE NEW RESPONSE WITH EARLY EXIT ########################
################################################################################
set_transformer_early_exit_mode(model, 'free_generate')
externalised_response, (externalised_generated_tokens, gathered_early_exit_layer_idxs) =\
    generate_text(model, prompt, system_prompt, prefiller, tokenizer, config['generation'], device)
print(externalised_response)


################################################################################
############################ GET SFT TEACHER ###################################
################################################################################
set_transformer_early_exit_mode(model, 'sft_teacher')
sft_teacher_response, (sft_teacher_generated_tokens, sft_teacher_final_layer_logprobs, gathered_early_exit_hidden_states) =\
    generate_text(model, prompt, system_prompt, prefiller, tokenizer, config['generation'], device)
print(sft_teacher_response)

early_output_log_probs = model.early_exit_hidden_state_readout(gathered_early_exit_hidden_states)
early_exit_probs = model.early_exit_target_probs(early_output_log_probs = early_output_log_probs, teacher_final_layer_log_probs = sft_teacher_final_layer_logprobs)


##############################################################################################
# XXX: SAMPLE EARLY EXIT PRESCRIPTIONS -- MULTIPLE FOR THE BATCH - COMPLETELY FAKE RIGHT NOW #
##### XXX: TO SORT OUT: slen IS LENGTH MINUS ONE - HOW DOES THAT LINE UP FOR SFT STUDENT #####
##############################################################################################
num_exit_samples = 4
sft_teacher_generated_tokens = sft_teacher_generated_tokens.repeat(num_exit_samples, 1)

# [samples, sequence length]
batch, slen, elayers = early_exit_probs.shape
assert batch == 1
sampled_early_exit_layer_idxs_early = torch.distributions.Categorical(probs = early_exit_probs[0]).sample((num_exit_samples,))
sampled_early_exit_layer_idxs = model.exitable_layer_idxs[sampled_early_exit_layer_idxs_early.cpu()]

################################################################################
################## GET SFT STUDENT OUTPUTS WITH PRESCRIBED EXITS ###############
################################################################################
set_transformer_early_exit_mode(model, 'sft_student')
output_scores, sft_student_info = model(sft_teacher_generated_tokens, prescribed_exit_layer_idxs = sampled_early_exit_layer_idxs)
