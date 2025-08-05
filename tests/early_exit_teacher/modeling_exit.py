import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict, Optional, Literal
import random
from dataclasses import dataclass
import math
from shared_utils.generate import format_conversation, transform_conversations
from tests.early_exit_teacher.visualization import create_html_visualization

@dataclass
class GenerationResult:
    """Container for generation results."""
    generated_tokens: List[int]
    token_strings: List[str]
    generated_text: str
    chosen_exit_layers: List[int]
    kl_divergences: Optional[List[torch.Tensor]] = None


class EarlyExitGenerator:
    """
    Unified class for text generation with early exit capabilities.
    
    Supports three modes:
    1. 'normal': Standard generation without early exit
    2. 'unfrozen': Early exit without freezing KV cache
    3. 'frozen': Early exit with KV cache freezing after exit
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_half: bool = True
    ):
        """
        Initialize the generator with a model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            load_in_half: Whether to load model in float16
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        dtype = torch.float16 if load_in_half and self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        
        # Extract early exit layer indices
        self.early_exit_layer_idxs = self._get_early_exit_layers()
        print(f"Early exit layers: {self.early_exit_layer_idxs.tolist()}")
        print(f"Total exitable layers: {len(self.early_exit_layer_idxs)}")
        self.LAST_LAYER_INDEX = self.model.config.num_hidden_layers - 1
        print(f"Last layer index: {self.LAST_LAYER_INDEX}")
    
    def _get_early_exit_layers(self) -> torch.Tensor:
        """Extract early exit layer indices from the model."""
        early_exit_layers = []
        
        # Check every 5th layer (as per module_name_is_layer_base logic)
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if len(parts) >= 2 and parts[-2] == 'layers':
                layer_idx = int(parts[-1])
                if layer_idx % 5 == 0:
                    early_exit_layers.append(layer_idx)
        
        return torch.tensor(early_exit_layers, dtype=torch.int32)
    
    def _calculate_kl_divergence(
        self,
        final_logits: torch.Tensor,
        early_exit_logits: torch.Tensor
    ) -> torch.Tensor:
        """Calculate KL divergence between final and early exit predictions."""
        final_probs = torch.softmax(final_logits, dim=-1)
        early_probs = torch.softmax(early_exit_logits, dim=-1)
        
        # Expand final probs for broadcasting
        final_probs_expanded = final_probs.unsqueeze(1)
        
        # KL divergence calculation
        eps = 1e-16
        kl_div = (final_probs_expanded * 
                  ((final_probs_expanded + eps) / (early_probs + eps)).log()).sum(-1)
        
        return kl_div
    
    def _apply_sigmoid_transformation(
        self,
        kl_div: torch.Tensor,
        kl_factor: float
    ) -> torch.Tensor:
        """Apply sigmoid transformation to KL divergence."""
        sigmoid_kls = torch.sigmoid(kl_factor * kl_div)
        sigmoid_kls = 2.0 * sigmoid_kls - 1.0
        sigmoid_kls = 1.0 - sigmoid_kls
        return sigmoid_kls
    
    def _determine_exit_layer(
        self,
        exit_probs: torch.Tensor,
        early_exit_logits: torch.Tensor,
        final_predictions: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """Determine which layer to exit from based on probabilities."""
        
        for idx, layer in enumerate(self.early_exit_layer_idxs):
            if random.random() < exit_probs[0, idx]:
                chosen_layer_idx = layer.item()
                return chosen_layer_idx, early_exit_logits[:, idx, :]
        
        # No early exit - return final layer
        return self.LAST_LAYER_INDEX, final_predictions
    
    def _generate_normal(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int
    ) -> GenerationResult:
        """Standard generation without early exit."""
        # import ipdb; ipdb.set_trace()
        current_input = input_ids.clone()
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = self.model(current_input, use_cache=True)
                else:
                    outputs = self.model(next_token, past_key_values = kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                
                if next_token.item() == eos_token_id:
                    break
                
                current_input = torch.cat([current_input, next_token], dim=1)
                generated_tokens.append(next_token.item())
        
        token_strings = [self.tokenizer.decode([t], skip_special_tokens=False) 
                        for t in generated_tokens]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return GenerationResult(
            generated_tokens=generated_tokens,
            token_strings=token_strings,
            generated_text=generated_text,
            chosen_exit_layers=[self.LAST_LAYER_INDEX] * len(generated_tokens)
        )
    
    def _generate_unfrozen(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        kl_factor: float = 1.0
    ) -> GenerationResult:
        """Generation with early exit but without freezing KV cache."""
        # import ipdb; ipdb.set_trace()
        current_input = input_ids.clone()
        generated_tokens = []
        chosen_exit_layers = []
        kl_divergences = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = self.model(current_input, use_cache=True)
                    # Get final layer logits
                    next_token_logits = outputs.logits[:, -1, :]
                    kv_cache = outputs.past_key_values
                else:
                    # Use previous outputs to get past key values     
                    outputs = self.model(
                        next_token,
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    kv_cache = outputs.past_key_values
                    
                    final_layer_logits = outputs.logits[:, -1, :]
                    hidden_states = torch.stack(outputs.hidden_states)[1:]  # Skip embedding layer to match the kv cache
                    exit_hidden_states = hidden_states[self.early_exit_layer_idxs, :, -1, :]
                    exit_hidden_states = exit_hidden_states.transpose(0, 1)
                    early_exit_logits = self.model.lm_head(exit_hidden_states) # Get early exit predictions
                    kl_div = self._calculate_kl_divergence(final_layer_logits, early_exit_logits)
                    exit_probs = self._apply_sigmoid_transformation(kl_div, kl_factor) # Apply sigmoid transformation
                    
                    # Determine exit layer
                    chosen_layer, next_token_logits = self._determine_exit_layer(
                        exit_probs, early_exit_logits, final_layer_logits
                    )
                
                # Sample next token (greedy)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                current_input = torch.cat([current_input, next_token], dim=1)
                generated_tokens.append(next_token.item())
                if step == 0:
                    chosen_exit_layers.append(self.LAST_LAYER_INDEX)
                    kl_divergences.append(None)
                else:
                    chosen_exit_layers.append(chosen_layer)  
                    kl_divergences.append(exit_probs[0]) # Calculate KL divergence              
                if next_token.item() == eos_token_id:
                    break
    
        token_strings = [self.tokenizer.decode([t], skip_special_tokens=False) 
                        for t in generated_tokens]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return GenerationResult(
            generated_tokens=generated_tokens,
            token_strings=token_strings,
            generated_text=generated_text,
            chosen_exit_layers=chosen_exit_layers,
            kl_divergences=kl_divergences
        )
    
    def _generate_frozen(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int,
        kl_factor: float = 1.0
    ) -> GenerationResult:
        """Generation with early exit and KV cache freezing."""
        current_input = input_ids.clone()
        generated_tokens = []
        chosen_exit_layers = []
        kl_divergences = []
        
        with torch.no_grad():
            # Initial forward pass
            initial_outputs = self.model(current_input, use_cache=True)
            student_kv_cache = initial_outputs.past_key_values
            # The following line may look redundant, but it initializes the teacher KV cache
            # for the first step. We can clone instead if needed.
            # This is important to ensure the teacher model has its own cache.
            initial_outputs = self.model(current_input, use_cache=True)
            teacher_kv_cache = initial_outputs.past_key_values
            
            for step in range(max_new_tokens):
                if step == 0:
                    # Use initial outputs
                    next_token = torch.argmax(initial_outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_tokens.append(next_token.item())
                    chosen_exit_layers.append(self.LAST_LAYER_INDEX)
                    kl_divergences.append(None)
                else:
                    # Student forward pass with early exit
                    student_outputs = self.model(
                        next_token,
                        past_key_values=student_kv_cache,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    
                    # Teacher forward pass to determine exit
                    early_exit_layer, teacher_kv_cache, exit_probs_teacher = self._determine_early_exit_from_teacher(
                        next_token, teacher_kv_cache, kl_factor
                    )
                    
                    # Get student predictions at chosen layer
                    student_hidden_states = torch.stack(student_outputs.hidden_states)[1:]
                    if early_exit_layer == self.LAST_LAYER_INDEX:
                        student_logits = student_outputs.logits
                    else:
                        student_logits = self.model.lm_head(student_hidden_states[early_exit_layer])
                    
                    # Update student KV cache - freeze at exit layer
                    student_kv_cache = student_outputs.past_key_values
                    if early_exit_layer != len(student_hidden_states) - 1:
                        for layer in range(early_exit_layer, len(student_kv_cache)):
                            # Freeze keys and values at the exit layer
                            student_kv_cache[layer][0][:, :, -1] = student_kv_cache[early_exit_layer][0][:, :, -1]
                            student_kv_cache[layer][1][:, :, -1] = student_kv_cache[early_exit_layer][1][:, :, -1]
                    
                    # Sample next token
                    next_token = torch.argmax(student_logits[:, -1, :], dim=-1).unsqueeze(-1)
                    generated_tokens.append(next_token.item())
                    chosen_exit_layers.append(early_exit_layer)
                    kl_divergences.append(exit_probs_teacher[0])
                
                if next_token.item() == eos_token_id:
                    break
        
        token_strings = [self.tokenizer.decode([t], skip_special_tokens=False) 
                        for t in generated_tokens]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return GenerationResult(
            generated_tokens=generated_tokens,
            token_strings=token_strings,
            generated_text=generated_text,
            chosen_exit_layers=chosen_exit_layers,
            kl_divergences=kl_divergences
        )
    
    def _determine_early_exit_from_teacher(
        self,
        next_token: torch.Tensor,
        teacher_kv_cache: Tuple,
        kl_factor: float
    ) -> Tuple[int, Tuple]:
        """Determine early exit layer using teacher model."""
        teacher_outputs = self.model(
            next_token,
            past_key_values=teacher_kv_cache,
            use_cache=True,
            output_hidden_states=True
        )
        
        # Get teacher predictions
        teacher_hidden_states = torch.stack(teacher_outputs.hidden_states)[1:]
        exit_hidden_states = teacher_hidden_states[self.early_exit_layer_idxs, :, -1, :]
        exit_hidden_states = exit_hidden_states.transpose(0, 1)
        
        teacher_logits = teacher_outputs.logits[:, -1, :]
        teacher_exit_logits = self.model.lm_head(exit_hidden_states)
        
        # Calculate KL divergence
        kl_div = self._calculate_kl_divergence(teacher_logits, teacher_exit_logits)
        
        # Apply sigmoid transformation
        exit_probs = self._apply_sigmoid_transformation(kl_div, kl_factor)
        
        # Determine exit layer
        chosen_layer, _ = self._determine_exit_layer(exit_probs, teacher_exit_logits, teacher_logits)
        
        return chosen_layer, teacher_outputs.past_key_values, exit_probs
    
    def generate(
        self,
        prompt: str,
        mode: Literal['normal', 'unfrozen', 'frozen'] = 'normal',
        max_new_tokens: int = 100,
        kl_factor: float = 1.0,
        system_prompt: Optional[str] = None,
        prefiller: str = ""
    ) -> GenerationResult:
        """
        Generate text using the specified mode.
        
        Args:
            prompt: Input prompt
            mode: Generation mode ('normal', 'unfrozen', or 'frozen')
            max_new_tokens: Maximum tokens to generate
            kl_factor: KL divergence scaling factor for early exit
            system_prompt: Optional system prompt
            prefiller: Optional prefiller text
            
        Returns:
            GenerationResult containing generated text and metadata
        """
        
        pre_transformed_conversation = format_conversation(user_prompts=[prompt], system_prompt=system_prompt)
        formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]
        # print(f"Formatted prompt: {formatted_prompt}")  
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        # print(f"Input IDs shape: {input_ids.shape}")
        
        # Get EOS token ID
        eos_token_id = self.tokenizer.eos_token_id
        
        # Generate based on mode
        if mode == 'normal':
            return self._generate_normal(input_ids, max_new_tokens, eos_token_id)
        elif mode == 'unfrozen':
            return self._generate_unfrozen(input_ids, max_new_tokens, eos_token_id, kl_factor)
        elif mode == 'frozen':
            return self._generate_frozen(input_ids, max_new_tokens, eos_token_id, kl_factor)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'normal', 'unfrozen', or 'frozen'.")
    
    def get_exit_layer_statistics(
        self,
        exit_layers: List[int]
    ) -> Dict[int, Dict[str, float]]:
        """Calculate statistics about exit layer usage."""
        total = len(exit_layers)
        if total == 0:
            return {}
        
        stats = {}
        all_layers = self.early_exit_layer_idxs.tolist() + [self.LAST_LAYER_INDEX]
        # import ipdb; ipdb.set_trace()
        for layer in all_layers:
            count = exit_layers.count(layer)
            stats[layer] = {
                'count': count,
                'percentage': (count / total) * 100,
                'layer_name': f"Layer {layer}" if layer != self.LAST_LAYER_INDEX else "Final Layer"
            }
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = EarlyExitGenerator(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device="cuda",
        load_in_half=False
    )
    
    # Test prompts
    test_prompts = [
        "Explain the concept of recursion in programming.",
        "What are the main causes of climate change?"
    ]
    system_prompt = "You are a helpful programming tutor."
    
    # Test all modes
    # modes = ['normal', 'unfrozen', 'frozen']
    # kl_factors = [1.0, 4.0]
    max_new_tokens = 100
    modes = ['normal', 'unfrozen', 'frozen']
    key_dict = {
        'normal': '1. Normal Generation',
        'unfrozen': '2. Early Exit',
        'frozen': '3. Early Exit + Freeze KV Cache'}
    kl_factors = [1.0]
    all_results = {key_dict[mode]: {} for mode in modes}
    for key_idx, mode in enumerate(modes):
        print(f"\n{'='*60}")
        print(f"Mode: {mode.upper()}")
        print(f"{'='*60}")
        key = key_dict[mode]
        for prompt_idx, prompt in enumerate(test_prompts):  # Just test one prompt for demo
            print(f"\nPrompt: {prompt}")            
            if mode == 'normal':
                result = generator.generate(
                    prompt=prompt,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    system_prompt = system_prompt
                )
                print(f"Generated: {result.generated_text}")
            else:
                for kl_factor in kl_factors:
                    print(f"\nKL Factor: {kl_factor}")
                    result = generator.generate(
                        prompt=prompt,
                        mode=mode,
                        max_new_tokens=max_new_tokens,
                        kl_factor=kl_factor,
                        system_prompt=system_prompt
                    )
                    
                    # Print results
                    print(f"Generated: {result.generated_text}")
                    
                    # # Print statistics
                    # stats = generator.get_exit_layer_statistics(result.chosen_exit_layers)
                    # print("\nExit layer distribution:")
                    # for layer, layer_stats in stats.items():
                    #     # if layer_stats['count'] > 0:
                    #     print(f"  {layer_stats['layer_name']}: "
                    #             f"{layer_stats['count']} tokens "
                    #             f"({layer_stats['percentage']:.1f}%)")
                    # Store results in format expected by visualization
            all_results[key][prompt_idx] = (
                        result.token_strings,
                        result.chosen_exit_layers,
                        result.generated_text,
                        result.kl_divergences if result.kl_divergences else [None] * len(result.token_strings)
                    )
    exit_layers_for_viz = []
    for i, layer in enumerate(generator.early_exit_layer_idxs):
        exit_layers_for_viz.append(layer.item())
        
    create_html_visualization(
        all_results=all_results,
        early_exit_layer_idxs=torch.tensor(exit_layers_for_viz),
        test_prompts=test_prompts,
        output_path='tests/early_exit_teacher/visualizations/early_exit_comparison.html',
        title='Early Exit Generation Modes Comparison'
    )
            