from torch import Tensor as _T

from types import MethodType

from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, TaskType, PeftModelForCausalLM

from transformers.utils import logging
logger = logging.get_logger(__name__)


from early_exit.util import *
from early_exit.patching.dynamical_types import generate_layer_type_with_early_exit_decision_head, generate_model_type_with_early_exit_readout_head
from early_exit.patching.attention_mixins.base import LayerFakeAttentionForwardMixin, possible_early_exit_types
from early_exit.patching.model_mixins.base import EarlyExitModelMixin



def patched_forward_generation(self: EarlyExitModelMixin | PeftModelForCausalLM, input_ids: _T, *args, **kwargs):
    """
    When in generation mode, the model should exit generation as soon as the ExitState
    is populated. This patched version of the attention block forward keeps a track of
    whether we are exitting early, by catching the custom error in generation_mode_hook_fn
    """
    if isinstance(self, PeftModelForCausalLM):
        raise NotImplementedError

    exit_state = ExitLogger(batch_size = input_ids.shape[0])

    assert self.early_exit_mode == 'free_generate'
    for name, module in self.named_modules():
        if module_name_is_layer_base(name):
            print("Free generate: Patched forward generation called at ", name)
            assert module.early_exit_mode == 'free_generate'
            module.exit_state = exit_state

    outputs = self.base_model_forward(input_ids, *args, **kwargs)

    self._early_exit_logs.append(exit_state)
    
    # No early exit occurred
    return outputs



def patched_forward_sft_teacher(self: EarlyExitModelMixin | PeftModelForCausalLM, input_ids: _T, *args, **kwargs):
    """
    When distilling to another model, the teacher never exits, but collects hidden states of exiting along the way

    These will be collected as gathered_early_exit_states of shape [batch, num exitable layers, generation length - 1, hidden dim]
    These are to be used to calculate probability_of_exit of shape [batch, generation length - 1, num exitable layers],
        to be sampled from and used as prescribed_exit_layer_idxs in patched_forward_sft_student
    
    input_ids of shape [batch, prompt length + generation length]
    """
    if isinstance(self, PeftModelForCausalLM):
        raise NotImplementedError

    batch_size, prompt_length = input_ids.shape

    exit_state = ExitProbabilities(batch_size=batch_size, hidden_size=self.config.hidden_size)

    for name, module in self.named_modules():
        if module_name_is_layer_base(name):
            assert module.early_exit_mode == 'sft_teacher'
            module.exit_state = exit_state

    self._early_exit_probabilities.append(exit_state)

    return self.base_model_forward(input_ids, *args, **kwargs)



def patched_forward_sft_student(self: EarlyExitModelMixin | PeftModelForCausalLM, input_ids: _T, *args, prescribed_exit_layer_idxs: _T, **kwargs):
    """
    When being distilled to, the model has prescribed exit points for each timestep.
        XXX: prescribed_exit_layer_idxs should be sampled upstream of this

    These are fed in as prescribed_exit_layer_idxs of shape [batch, generation length], which given layer_idx 
    
    input_ids of shape [batch, prompt length + generation length]
    """
    if isinstance(self, PeftModelForCausalLM):
        raise NotImplementedError

    assert self.early_exit_mode == 'sft_student'

    exit_state = ExitPrescription(
        prescribed_exit_layer_idxs = prescribed_exit_layer_idxs,
        total_exitable_layers = self.total_exitable_layers,
        device = input_ids.device
    )

    batch_size, total_length = input_ids.shape
    assert batch_size == exit_state.batch_size
    assert total_length > exit_state.generation_length
    
    assert self.early_exit_mode == 'sft_student'
    for name, module in self.named_modules():
        if module_name_is_layer_base(name):
            assert module.early_exit_mode == 'sft_student'
            module.exit_state = exit_state

    return self.base_model_forward(input_ids, *args, **kwargs), exit_state.collected_exit_logits



def patched_generate(self: EarlyExitModelMixin | PeftModelForCausalLM, *args, **kwargs):
    """
    As well as generating the text rollout as in the base class,
    we also want to collate the ExitStates which were produced by patched_forward_generation

    gathered_early_exit_layer_idxs output as shape [batch, generation length]
    """
    if isinstance(self, PeftModelForCausalLM):
        raise NotImplementedError
    if self.early_exit_mode == 'off':
        return self.base_model_generate(*args, **kwargs)

    elif self.early_exit_mode == 'free_generate':
        self._early_exit_logs: List[ExitLogger] = []
        outputs = self.base_model_generate(*args, **kwargs)
        gathered_early_exit_layer_idxs = [ee.readout_layer_idx for ee in self._early_exit_logs]
        gathered_early_exit_layer_idxs = torch.tensor(gathered_early_exit_layer_idxs).T
        assert gathered_early_exit_layer_idxs[:,0].isinf().all()
        self._early_exit_logs = None
        return outputs, gathered_early_exit_layer_idxs

    elif self.early_exit_mode == 'sft_teacher':
        # XXX: final_layer_logprobs extra pmf cut off correctly?
        self._early_exit_probabilities: List[ExitProbabilities] = []
        outputs = self.base_model_generate(*args, **kwargs, return_dict_in_generate = True, output_scores = True)
        final_layer_logprobs = torch.stack(outputs.scores, 1).log_softmax(-1)[:,1:]       # [B, generation length - 1, vocab]
        assert self._early_exit_probabilities[0].cache == []
        gathered_early_exit_states = torch.concat([torch.stack(ee.cache, 1) for ee in self._early_exit_probabilities[1:]], 2)
        self._early_exit_probabilities = None
        return outputs.sequences, final_layer_logprobs, gathered_early_exit_states

    else:
        raise Exception(f'cannot use patched_generate when early_exit_mode == {self.early_exit_mode}')



def set_layer_early_exit_mode(module: LayerFakeAttentionForwardMixin, mode: possible_early_exit_types):

    module.self_attn.forward = (
        module.self_attn.patched_self_attn_forward if mode == 'free_generate' else      # Can skip
        module.self_attn.patched_self_attn_forward if mode == 'sft_student' else        # Forced to skip
        module.self_attn.base_self_attn_forward if mode == 'sft_teacher' else           # Can't skip
        module.self_attn.base_self_attn_forward if mode == 'off' else                   # Base model
        None
    )

    module.early_exit_mode = mode



def set_transformer_early_exit_mode(model: EarlyExitModelMixin | PeftModelForCausalLM, mode: possible_early_exit_types):
    if isinstance(model, PeftModelForCausalLM):
        assert isinstance(model.base_model.model, EarlyExitModelMixin)
        set_transformer_early_exit_mode(model = model.base_model.model, mode = mode)

    if mode in ['off', 'sft_teacher']:
        model.disable_adapters()
    elif mode in ['free_generate', 'sft_student']:
        model.enable_adapters()
    else:
        raise ValueError

    if mode == 'sft_student':
        model.train()
    else:
        model.eval()

    for name, module in model.named_modules():
        if module_name_is_layer_base(name):
            set_layer_early_exit_mode(module, mode)
            

    model.forward = (
        model.patched_forward_generation if mode == 'free_generate' else
        model.patched_forward_sft_student if mode == 'sft_student' else
        model.patched_forward_sft_teacher if mode == 'sft_teacher' else
        model.base_model_forward if mode == 'off' else
        None
    )

    model.early_exit_mode = mode



def replace_attention_layers(model: AutoModelForCausalLM, lora_config_dict: dict, device: str = 'cuda') -> EarlyExitModelMixin:
    """
    Replace attention layers with augmented versions
    """
    exitable_layer_idx = 0
    exitable_layer_idxs = []

    new_model_class = generate_model_type_with_early_exit_readout_head(type(model))
    new_model = new_model_class(model.config)
    new_model.load_state_dict(model.state_dict())
    del model
    model: new_model_class = new_model

    for name, module in model.named_modules():

        if module_name_is_layer_base(name):

            augmented_type = generate_layer_type_with_early_exit_decision_head(base_type = type(module))

            # XXX: should be a more robust way to extract config and layer_idx
            new_layer = augmented_type(
                config = module.self_attn.config,
                layer_idx = module.self_attn.layer_idx,
                exitable_layer_idx = exitable_layer_idx,
            )

            exitable_layer_idxs.append(new_layer.layer_idx)

            load_state = new_layer.load_state_dict(module.state_dict(), strict=False)
            # assert load_state.missing_keys == ['early_exit_decision_weights.weight', 'early_exit_decision_weights.bias', 'early_readout_weights.weight'], load_state.missing_keys
            assert load_state.missing_keys == ['early_exit_decision_weights.weight', 'early_exit_decision_weights.bias'], load_state.missing_keys

            parent = dict(model.named_modules())[name.rsplit('.', 1)[0]]
            setattr(parent, name.rsplit('.', 1)[-1], new_layer.to(device = model.device, dtype=model.dtype))

            print(f'replacing layer {name}')
            exitable_layer_idx += 1

    model.base_model_forward = model.forward  # Keep original
    model.patched_forward_generation = MethodType(patched_forward_generation, model)
    model.patched_forward_sft_student = MethodType(patched_forward_sft_student, model)
    model.patched_forward_sft_teacher = MethodType(patched_forward_sft_teacher, model)
    
    model.base_model_generate = model.generate
    model.generate = MethodType(patched_generate, model)

    model.total_exitable_layers = exitable_layer_idx
    model.exitable_layer_idxs = torch.tensor(exitable_layer_idxs + [float('inf')])

    print('address this hack!')
    model._hf_peft_config_loaded = True

    lora_config = LoraConfig(**lora_config_dict, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config, adapter_name="early_exiter")

    model.set_adapter("early_exiter")
    model.print_trainable_parameters()
    model.enable_adapters()

    set_transformer_early_exit_mode(model, 'off')
    
    model.requires_grad_(False)  # Freeze all
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        if 'early_exit_decision_weights' in name:
            param.requires_grad = True

    return model.to(device)