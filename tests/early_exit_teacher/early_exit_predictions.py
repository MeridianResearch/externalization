import torch 
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from tests.early_exit_teacher.visualization import safe_decode_tokens, visualize_tokens_by_exit_layer
from early_exit.patching.method_patching import replace_attention_layers
from shared_utils.load import get_tokenizer, configs_from_yaml
from early_exit.util import get_model, module_name_is_layer_base
from shared_utils.generate import format_conversation, transform_conversations
from typing import Callable, Optional, List, Tuple
from IPython.display import display
import numpy as np
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
import os, certifi
api_key = os.environ["OPENAI_API_KEY"]
os.environ["SSL_CERT_FILE"] = certifi.where()
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import scorer, accuracy, Score
from inspect_ai.solver import system_message
from inspect_ai.model import get_model as get_inspect_model

async def evaluate_response(tokenizer, generated_tokens, chosen_exit_layers, prompt, exitable_layers):
    """
    Evaluate the quality and coherence of a generated response using GPT-5 as a judge.
    
    Args:
        generated_tokens: List of generated token IDs
        chosen_exit_layers: List of exit layers used for each token
        prompt: The original input prompt
        system_prompt: System prompt used (if any)
    
    Returns:
        Dict containing evaluation scores and metrics
    """
    
    # Decode the response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Calculate early exit statistics
    total_tokens = len(generated_tokens)
    finite_exits = [layer for layer in chosen_exit_layers if layer not in [27, -1, float('inf')]]
    early_exits = len(finite_exits)
    early_exit_rate = early_exits / total_tokens if total_tokens > 0 else 0
    usage = np.mean(np.array(chosen_exit_layers)/27.0)
    
    # Calculate layer distribution
    layer_distribution = {}
    if finite_exits:
        unique_layers, counts = np.unique(finite_exits, return_counts=True)
        for layer, count in zip(unique_layers, counts):
            layer_distribution[int(layer)] = int(count)
    
    # Prepare evaluation prompt
    eval_prompt = f"""
You are evaluating the coherence and quality of an AI assistant's response.

Original Question: {prompt}

AI Response: {response}

Please evaluate this response on the following criteria:
1. Coherence and logical flow (1-10 scale)
2. Completeness of the answer (1-10 scale)  
3. Clarity and readability (1-10 scale)
4. Absence of repetition or errors (1-10 scale)

Rate each criterion and provide an overall score from 1-10:
- 1: Poor (major issues)
- 4: Fair (some issues)
- 7: Good (minor issues)
- 10: Excellent (no issues)

Format your response as:
Coherence: X/10
Completeness: X/10
Clarity: X/10
No Repetition: X/10
Overall: X/40

Brief explanation: [your reasoning]
"""
    
    try:
        # Get GPT-5 model for evaluation
        judge_model = get_inspect_model("openai/gpt-5")
        
        # Generate evaluation
        eval_result = await judge_model.generate(eval_prompt)
        eval_text = eval_result.completion
        
        # Parse scores from the evaluation text
        scores = {
            'coherence': 0,
            'completeness': 0,
            'clarity': 0,
            'no_repetition': 0,
            'overall': 0
        }
        
        for line in eval_text.split('\n'):
            if 'Coherence:' in line:
                try:
                    scores['coherence'] = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif 'Completeness:' in line:
                try:
                    scores['completeness'] = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif 'Clarity:' in line:
                try:
                    scores['clarity'] = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif 'No Repetition:' in line:
                try:
                    scores['no_repetition'] = int(line.split(':')[1].strip().split('/')[0])
                except:
                    pass
            elif 'Overall:' in line:
                try:
                    score_part = line.split(':')[1].strip()
                    scores['overall'] = int(score_part.split('/')[0])
                except:
                    pass
        
        # Convert overall score to 0-1 scale
        accuracy_score = scores['overall'] / 40.0 if scores['overall'] > 0 else 0
        
        # Prepare results
        evaluation_results = {
            'response': response,
            'scores': scores,
            'accuracy_score': accuracy_score,
            'early_exit_stats': {
                'total_tokens': total_tokens,
                'early_exits': early_exits,
                'early_exit_rate': early_exit_rate,
                'layer_distribution': layer_distribution,
                'usage': usage
            },
            'evaluation_text': eval_text,
            'chosen_exit_layers': [27 if item == 27 or item == -1 else item for item in chosen_exit_layers],
            'tokens': safe_decode_tokens(tokenizer, generated_tokens),
            'exitable_layers': exitable_layers
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"\nEarly Exit Statistics:")
        print(f"  Exit Rate: {early_exit_rate:.1%} ({early_exits}/{total_tokens} tokens)")
        print(f"  Layer Distribution: {layer_distribution}")
        print(f"\nQuality Scores:")
        print(f"  Coherence: {scores['coherence']}/10")
        print(f"  Completeness: {scores['completeness']}/10")
        print(f"  Clarity: {scores['clarity']}/10")
        print(f"  No Repetition: {scores['no_repetition']}/10")
        print(f"  Overall: {scores['overall']}/40 (Accuracy: {accuracy_score:.2f})")
        print(f"\nExplanation: {eval_text.split('Brief explanation:')[-1].strip() if 'Brief explanation:' in eval_text else 'N/A'}")
        print(f"{'='*60}\n")
        
        return evaluation_results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Return basic statistics even if evaluation fails
        return {
            'chosen_exit_layers': [27 if item == 27 or item == -1 else item for item in chosen_exit_layers],
            'tokens': safe_decode_tokens(tokenizer, generated_tokens),
            'response': response,
            'scores': None,
            'accuracy_score': None,
            'early_exit_stats': {
                'total_tokens': total_tokens,
                'early_exits': early_exits,
                'early_exit_rate': early_exit_rate,
                'layer_distribution': layer_distribution
            },
            'evaluation_text': f"Evaluation failed: {str(e)}"
        }


def load_default_model_and_tokenizer(model_config_path = "../../config_deepseek.yaml"):
    # Model configuration
    device = 'cuda' 
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = get_tokenizer(model_name)
    config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)

    model = get_model(model_name, config['model'], device)
    model = replace_attention_layers(model, config['lora'], device)
    # set_transformer_early_exit_mode(model, 'off')

    # Load tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
def format_and_tokenize_input(prompt, system_prompt, prefiller, tokenizer, device):
    pre_transformed_conversation = format_conversation(user_prompts = [prompt], system_prompt=system_prompt)
    formatted_prompt = transform_conversations(pre_transformed_conversation, prefiller)[0]
    return tokenizer(formatted_prompt, return_tensors="pt").to(device)

def get_early_exit_indices(model):
    early_exit_layer_idxs = []
    for name, module in model.named_modules():
        if module_name_is_layer_base(name):
            layer_idx = int(name.split('.')[-1])
            early_exit_layer_idxs.append(layer_idx)

    return torch.tensor(early_exit_layer_idxs, dtype = torch.int32)  # Add inf for final layer

class PredictionObject:
    """
    Stores the state of the current sequence generation, including
    cache, logits, and generated tokens at each step.
    """
    def __init__(self, model):
        self.model = model
        self.initial_prompt = []
        self.generated_tokens = [] # This is not strictly the generated tokens, but more like the tokens passed in do NTP
        self.all_tokens = []
        self.chosen_exit_layers = []
        self.all_logits = [] # stores only the prediction logits
        self.device = model.device
        self.cache = None  # To store past_key_values
    
    def build_cache(self, generated):
        """
        Update the prediction state for a new generation step.
        """
        outputs = self.model(generated, use_cache=True)
        # self.all_logits = outputs.logits
        self.cache = outputs.past_key_values
        return outputs.logits

    def get_early_exit_predictions(self, hidden_states, next_token, early_exit_layer, mode):
        exit_hidden_state = hidden_states[early_exit_layer]
        logits = self.model.lm_head(exit_hidden_state)
        self.all_logits.append(logits)
        assert len(self.cache) == hidden_states.shape[0]
        for layer_idx in range(early_exit_layer + 1, len(self.cache)):
            if mode == 'unfrozen':
                continue
            elif mode == 'frozen_cache':
                self.cache[layer_idx][0][:, :, -1] = self.cache[early_exit_layer][0][:, :, -1] # keys
                self.cache[layer_idx][1][:, :, -1] = self.cache[early_exit_layer][1][:, :, -1] # values
            elif mode == 'scrambled_cache_lot_of_noise':
                self.scramble_values(layer_idx, early_exit_layer, noise_scale=100)
            elif mode == 'scrambled_cache_little_of_noise':
                self.scramble_values(layer_idx, early_exit_layer, noise_scale=0.1)
            elif mode == 'frozen_residual':
                layer = self.model.base_model.model.model.layers[layer_idx]
                normed_hidden = layer.input_layernorm(exit_hidden_state)
    
                # Project to K and V using this layer's projections
                key_states = layer.self_attn.k_proj(normed_hidden)
                value_states = layer.self_attn.v_proj(normed_hidden)
                
                # Reshape for multi-head attention
                num_key_value_heads = layer.self_attn.config.num_key_value_heads
                head_dim = layer.self_attn.head_dim
                # print(key_states.shape, num_key_value_heads)
                key_states = key_states.view(1, 1, num_key_value_heads, head_dim).transpose(1, 2)
                value_states = value_states.view(1, 1, num_key_value_heads, head_dim).transpose(1, 2)
                # print(student_cache[0][0].shape)
                current_position = self.cache[0][0].shape[-2]
                position_ids = torch.tensor([[current_position]], device=self.device)
                cos, sin = self.model.base_model.model.model.rotary_emb(value_states, position_ids)
                _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                self.cache[layer_idx][0][:, :, -1:] = key_states # keys
                self.cache[layer_idx][1][:, :, -1:] = value_states # values
                
            else:
                raise ValueError(f"Unknown cache update mode: {mode}")
            
        return logits
    
    def update_cache(self, next_token, early_exit_layer, mode):
        outputs = self.model(
            next_token,
            past_key_values=self.cache, # the update of self.cache happens in-place
            use_cache=True,
            output_hidden_states=early_exit_layer is not None
        )
        if early_exit_layer is None:
            self.all_logits.append(outputs.logits)
            return outputs.logits
        else:
            hidden_states = torch.stack(outputs.hidden_states)[1:]
            logits = self.get_early_exit_predictions(hidden_states, next_token, early_exit_layer, mode)
            return logits
        
    def update_after_prediction(self, next_token, chosen_exit_layer):
        self.all_tokens.append(next_token)
        self.generated_tokens.append(next_token)
        self.chosen_exit_layers.append(chosen_exit_layer)

    
    def scramble_values(self, layer_idx, early_exit_layer, noise_scale):
        """
        Add noise to KV cache values for layers after early exit.
        """
        # Get the shape from the early exit layer's KV cache
        key_shape = self.cache[early_exit_layer][0][:, :, -1:].shape
        value_shape = self.cache[early_exit_layer][1][:, :, -1:].shape
        
        # Generate noise with the same shape and add to cache
        key_noise = torch.randn_like(self.cache[early_exit_layer][0][:, :, -1:]) * noise_scale
        value_noise = torch.randn_like(self.cache[early_exit_layer][1][:, :, -1:]) * noise_scale
        
        # Add noise to the cache at the specified layer
        self.cache[layer_idx][0][:, :, -1:] = self.cache[early_exit_layer][0][:, :, -1:] + key_noise
        self.cache[layer_idx][1][:, :, -1:] = self.cache[early_exit_layer][1][:, :, -1:] + value_noise
    

    def __repr__(self):
        return f"PredictionObject(len={len(self.generated_tokens)})"
    
    def update_teacher_cache(self, next_token):
        outputs = self.model(
            next_token,
            past_key_values=self.cache, # the update of self.cache happens in-place
            use_cache=True,
            output_hidden_states=True
        )
        return outputs
        
class EarlyExitGenerator:
    def __init__(self, model, tokenizer, mode, exitable_layers):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.exitable_layers = exitable_layers
    
    def set_exit_strategy(self, strategy_fn: Callable):
        self.set_early_exit_layer = strategy_fn
        
    def generate(self, inputs, max_new_tokens = 100, visualize_early_exit = False):
        prediction = PredictionObject(self.model)    
        input_ids = inputs["input_ids"].clone()
        for step in range(max_new_tokens):  # generate 10 tokens
            if step == 0:
                early_exit_layer = 27
                logits = prediction.build_cache(input_ids)
                # teacher_prediction.build_cache(model, generated)
            else:
                early_exit_layer = self.set_early_exit_layer()
                logits = prediction.update_cache(next_token, mode = self.mode, early_exit_layer = early_exit_layer)
                # teacher_prediction.update_cache(next_token)
            # Take the most likely next token (greedy decoding here)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            prediction.update_after_prediction(next_token.item(), early_exit_layer)

        generated_tokens = prediction.generated_tokens
        chosen_exit_layers = prediction.chosen_exit_layers
        assert len(chosen_exit_layers) == len(generated_tokens), \
            f"Mismatch: {len(chosen_exit_layers)} exit layers vs {len(generated_tokens)} tokens"
        
        self.print_generation(generated_tokens, chosen_exit_layers, visualize_early_exit = visualize_early_exit)
        if self.mode == 'frozen_cache':
            self._verify_frozen_cache(prediction, inputs, chosen_exit_layers)
            
        return generated_tokens, chosen_exit_layers, prediction
    
    def print_generation(self, generated_tokens, chosen_exit_layers, visualize_early_exit = False):
        print(self.tokenizer.decode(generated_tokens))
        if visualize_early_exit:
            tokens = safe_decode_tokens(self.tokenizer, generated_tokens)
            layers = [27 if item == 27 or item == -1 else item for item in chosen_exit_layers]
            display(visualize_tokens_by_exit_layer(tokens, layers, self.exitable_layers.tolist(), 
                                                   title="Committed Early Exit Token Generation"))
    
    
    def evaluate_response(self, generated_tokens, chosen_exit_layers, inputs):
        """
        Synchronous wrapper to evaluate the response using the async evaluate_response function.
        
        Args:
            generated_tokens: List of generated token IDs
            chosen_exit_layers: List of exit layers used for each token
            input_ids: The input token IDs (tensor)
        
        Returns:
            Dict containing evaluation scores and metrics
        """
        import asyncio
        input_ids = inputs['input_ids']
        # Decode the prompt from input_ids
        # Handle both batched and unbatched inputs
        if len(input_ids.shape) == 2:
            prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        else:
            prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Create or get event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create new event loop if none exists or current one is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async evaluation function
        try:
            eval_results = loop.run_until_complete(
                evaluate_response(self.tokenizer, generated_tokens, chosen_exit_layers, prompt, self.all_exitable_layers.tolist() if hasattr(self, 'all_exitable_layers') else self.exitable_layers.tolist())
            )
            return eval_results
        except Exception as e:
            print(f"Error during evaluation: {e}")
            # Return basic results if evaluation fails
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            total_tokens = len(generated_tokens)
            finite_exits = [layer for layer in chosen_exit_layers if layer not in [27, -1, float('inf')]]
            early_exits = len(finite_exits)
            early_exit_rate = early_exits / total_tokens if total_tokens > 0 else 0
            
            return {
                'response': response,
                'scores': None,
                'accuracy_score': None,
                'early_exit_stats': {
                    'total_tokens': total_tokens,
                    'early_exits': early_exits,
                    'early_exit_rate': early_exit_rate,
                    'layer_distribution': {}
                },
                'evaluation_text': f"Evaluation failed: {str(e)}"
            }

    def _verify_frozen_cache(self, prediction, inputs, chosen_exit_layers):
        """
        Verify that the cache is properly frozen for layers after early exit.
        This test ensures that for each token, all layers after the chosen exit layer
        have identical KV cache values to the exit layer.
        """
        gen_len = len(chosen_exit_layers)
        input_cache_len = inputs['input_ids'].shape[-1] - 1 # The -1 is a bit tricky here.
        
        for token_index in range(len(chosen_exit_layers)):
            chosen_layer = chosen_exit_layers[token_index]
            cache_position = input_cache_len + token_index
            
            # Skip if this was a full forward pass (no early exit)
            if chosen_layer == 27 or chosen_layer == -1:
                continue
                
            # Check all layers after the chosen exit layer
            for layer_idx in range(chosen_layer + 1, 28):
                # Verify key cache
                assert (prediction.cache[layer_idx][0][:, :, cache_position] == 
                    prediction.cache[chosen_layer][0][:, :, cache_position]).all(), \
                    f"Layer {layer_idx} key cache mismatch at token {token_index}, chosen layer = {chosen_layer}"
                
                # Verify value cache
                assert (prediction.cache[layer_idx][1][:, :, cache_position] == 
                    prediction.cache[chosen_layer][1][:, :, cache_position]).all(), \
                    f"Layer {layer_idx} value cache mismatch at token {token_index}, chosen layer = {chosen_layer}"
        
        print(f"\n✓ Cache verification passed for mode = {self.mode}: All {gen_len} tokens have properly frozen caches")
        
        def evaluate_response(self, generated_tokens):     
            # Here we would call the async evaluation function defined earlier
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(evaluate_response(self, generated_tokens, self.chosen_exit_layers, self.prompt, self.system_prompt))
            return result
        
class KLExitGenerator(EarlyExitGenerator):
    def __init__(self, model, tokenizer, mode, exitable_layers):
        super().__init__(model, tokenizer, mode, exitable_layers)
        self.all_exitable_layers = torch.cat([
            exitable_layers,
            torch.tensor([27], device=exitable_layers.device, dtype=exitable_layers.dtype),
        ])
    """
    Generator that uses KL divergence between early and final layer outputs
    to determine when to exit early.
    """
    def generate_normal(self, inputs, max_new_tokens = 100, visualize_early_exit = False):
        prediction = PredictionObject(self.model)    
        input_ids = inputs["input_ids"].clone()
        for step in range(max_new_tokens):  # generate 10 tokens
            if step == 0:
                early_exit_layer = 27
                logits = prediction.build_cache(input_ids)
            else:
                early_exit_layer = 27
                logits = prediction.update_cache(next_token, mode = self.mode, early_exit_layer = None)
            # Take the most likely next token (greedy decoding here)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            prediction.update_after_prediction(next_token.item(), early_exit_layer)
        generated_tokens = prediction.generated_tokens
        chosen_exit_layers = prediction.chosen_exit_layers
        assert len(chosen_exit_layers) == len(generated_tokens), \
            f"Mismatch: {len(chosen_exit_layers)} exit layers vs {len(generated_tokens)} tokens"
        return generated_tokens, chosen_exit_layers, prediction
    
    def generate(self, inputs, kl_strength, max_new_tokens = 100, visualize_early_exit = False):
        if self.mode == 'normal':
            return self.generate_normal(inputs, max_new_tokens, visualize_early_exit)
        student_prediction = PredictionObject(self.model)    
        teacher_prediction = PredictionObject(self.model)    
        input_ids = inputs["input_ids"].clone()
        for step in range(max_new_tokens):  # generate 10 tokens
            if step == 0:
                early_exit_layer = 27
                student_logits = student_prediction.build_cache(input_ids)
                teacher_prediction.build_cache(input_ids)
                # teacher_prediction.build_cache(model, generated)
            else:
                teacher_outputs = teacher_prediction.update_teacher_cache(next_token)
                hidden_states = torch.stack(teacher_outputs.hidden_states)[1:]
                early_output_log_probs = self.model.lm_head(hidden_states[self.exitable_layers]).squeeze().unsqueeze(0).unsqueeze(2).log_softmax(-1)  
                teacher_final_layer_log_probs = teacher_outputs.logits.squeeze().unsqueeze(0).unsqueeze(0).log_softmax(-1)
                early_exit_probs = self.model.early_exit_target_probs(
                    early_output_log_probs = early_output_log_probs, 
                    teacher_final_layer_log_probs = teacher_final_layer_log_probs,
                    KL_FACTOR = kl_strength
                    )
                early_exit_idx = torch.distributions.Categorical(probs = early_exit_probs).sample((1,)).item()     # [samples, batch, generation length] 
                early_exit_layer = self.all_exitable_layers[early_exit_idx].item()
                student_logits = student_prediction.update_cache(next_token, mode = self.mode, early_exit_layer = early_exit_layer)
            # Take the most likely next token (greedy decoding here)
            next_token = torch.argmax(student_logits[:, -1, :], dim=-1).unsqueeze(-1)
            student_prediction.update_after_prediction(next_token.item(), early_exit_layer)
            teacher_prediction.update_after_prediction(next_token.item(), 27)  # Teacher always uses final layer

        generated_tokens = student_prediction.generated_tokens
        chosen_exit_layers = student_prediction.chosen_exit_layers
        assert len(chosen_exit_layers) == len(generated_tokens), \
            f"Mismatch: {len(chosen_exit_layers)} exit layers vs {len(generated_tokens)} tokens"
        
        self.print_generation(generated_tokens, chosen_exit_layers, visualize_early_exit = visualize_early_exit)
        if self.mode == 'frozen_cache':
            self._verify_frozen_cache(student_prediction, inputs, chosen_exit_layers)
            
        return generated_tokens, chosen_exit_layers, student_prediction
    
    
        
    
    
