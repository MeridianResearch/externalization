import math

import torch
from torch import Tensor as _T

from dataclasses import dataclass, field
from typing import Optional, List

from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from pathlib import Path
import json
from peft import PeftModel
import wandb
import os


def module_name_is_layer_base(name: str):
    split_by_dots = name.split('.')
    if len(split_by_dots) < 2:
        return False
    is_layer = (split_by_dots[-2] == 'layers')
    if is_layer:
        layer_idx = int(split_by_dots[-1])
        return layer_idx % 5 == 0 and layer_idx > 0
        #return layer_idx % 3 == 0
        #return layer_idx >= 0
    else:
        return False

def module_name_is_transformer_layer(name: str) -> bool:
    """
    Check if a module name corresponds to a transformer layer.
    
    Args:
        name: Module name from model.named_modules()
        
    Returns:
        bool: True if the module is a transformer layer, False otherwise
    """
    split_by_dots = name.split('.')
    if len(split_by_dots) < 2:
        return False
    return split_by_dots[-2] == 'layers'


@dataclass
class ExitLogger:
    """
    Message carrier for exiting early, and the logits associated with an early exit
    """
    batch_size: int
    readout_layer_idx: Optional[List[int | float]] = None

    def __post_init__(self) -> None:
        self.readout_layer_idx = [torch.inf for _ in range(self.batch_size)]
        self.frozen_batch_items = []
    
    def freeze_batch_item(self, batch_idx: List[int], layer_idx: int) -> None:
        for bi in batch_idx:
            assert bi not in self.frozen_batch_items
            assert math.isinf(self.readout_layer_idx[bi])
            self.frozen_batch_items.append(bi)
            self.readout_layer_idx[bi] = layer_idx
        self.frozen_batch_items = sorted(self.frozen_batch_items)

    @property
    def unfrozen_batch_items(self) -> List[int]:
        return [i for i in range(self.batch_size) if i not in self.frozen_batch_items]


@dataclass
class ExitProbabilities:
    """
    This is for the teacher model in SFT

    Keeps a track of early hidden states, which can later be converted to logits *as if* we had exited early
    """
    batch_size: int
    hidden_size: int
    cache: List[_T] = field(default_factory=lambda: [])
    layers_idxs: List[int] = field(default_factory=lambda: [])
    exitable_layers_idxs: List[int] = field(default_factory=lambda: [])

    def log_residual_stream(self, hidden_states: _T, layer_idx: int, exitable_layer_idx: int) -> None:
        """
        Expecting hidden_states of shape [batch, length = 1, hidden size]
        
        Expecting length of 1 because this is only for 'freely generated' hidden states
        """
        assert tuple(hidden_states.shape) == (self.batch_size, 1, self.hidden_size)
        assert exitable_layer_idx == len(self.exitable_layers_idxs)
        
        self.cache.append(hidden_states)
        self.layers_idxs.append(layer_idx)
        self.exitable_layers_idxs.append(exitable_layer_idx)






@dataclass
class ExitPrescription:
    """
    This is for the student model in SFT

    Keeps a track of the probability of exiting, but only exit at the preprescribed time!
    """
    prescribed_exit_layer_idxs: _T
    total_exitable_layers: int
    device: str
    
    batch_size: Optional[int] = None
    generation_length: Optional[int] = None
    collected_exit_logits: Optional[_T] = None

    def __post_init__(self,):
        self.batch_size, self.generation_length = self.prescribed_exit_layer_idxs.shape
        self.collected_exit_logits = torch.nan * torch.ones(
            self.batch_size, self.generation_length, self.total_exitable_layers, device = self.device
        )

    def get_unfrozen_mask(self, current_layer_idx: int) -> _T:
        """
        Layer is frozen if prescribed_exit_layer_idxs (where the early exit happened during generation)
            is XXX: >= the current layer
        """
        return self.prescribed_exit_layer_idxs >= current_layer_idx

def get_model(model_name: str, model_config: dict, device: str):
    
    quantization_config = {
        "4bits": BitsAndBytesConfig(load_in_4bit=True),
        "8bits": BitsAndBytesConfig(load_in_8bit=True)
    }.get(model_config['load_precision_mode'], None)
    
    # Handle attention implementation if specified
    config = None
    if 'attn_implementation' in model_config:
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = model_config['attn_implementation']
    
    if model_config['lora']:
        raise NotImplementedError
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            config=config,
        )
        
    return model.to(device)

def save_model(model: PeftModel, save_path: str, upload_to_wandb: bool = True) -> None:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    #save LoRA adapters
    model.save_pretrained(save_path, save_embedding_layers=True)
    
    #save early exit decision weights
    probe_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'early_exit_decision_weights'):
            probe_weights[name] = module.early_exit_decision_weights.state_dict()
    
    if probe_weights:
        torch.save(probe_weights, save_path / "early_exit_probes.pt")
    
    # Save metadata
    metadata = {
        "base_model_name": getattr(model.base_model.model.config, 'name_or_path', 'unknown'),
        "exitable_layer_idxs": model.exitable_layer_idxs.tolist(),
        "total_exitable_layers": model.total_exitable_layers,
        "has_early_exit_probes": len(probe_weights) > 0
    }
    
    with open(save_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    if upload_to_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"early-exit-model-{wandb.run.id}",
            type="model",
            description="Early exit model with LoRA adapters"
        )
        artifact.add_dir(str(save_path))
        wandb.log_artifact(artifact)
        print("Model uploaded to WandB")
    elif upload_to_wandb:
        print("WandB run not active, skipping upload")

def load_model(model, model_path):
    adapter_path = os.path.join(model_path, "early_exiter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model.load_adapter(adapter_path, "early_exiter")
        model.set_adapter("early_exiter")
    else:
        model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=model.device))
    
    #early exit probe weights
    probe_weights_path = os.path.join(model_path, "early_exit_probes.pt")
    if os.path.exists(probe_weights_path):
        probe_weights = torch.load(probe_weights_path, map_location=model.device)
        loaded_probes = 0
        for name, module in model.named_modules():
            if name in probe_weights and hasattr(module, 'early_exit_decision_weights'):
                module.early_exit_decision_weights.load_state_dict(probe_weights[name])
                loaded_probes += 1
    
    return model

def load_model_from_wandb(model, model_path, artifact_path):

    api = wandb.Api()
    
    # Get the artifact
    artifact = api.artifact(artifact_path)
    artifact.download(root=model_path)
    model = load_model(model, model_path)
    
    return model