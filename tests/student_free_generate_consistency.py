#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from typing import Optional, Dict, Any

# Import your custom modules
from early_exit.util import get_model, load_model_from_wandb, load_model
from early_exit.rl_utils import (
    apply_masking, generate_k_completions, center_rewards_per_prompt, 
    map_layers_to_indices, weighted_sft_step, get_input_prompt_length
)
from early_exit.rl_types import RLHyperparams, RolloutBatch
from early_exit.rewards import (
    compute_verification_rewards, compute_token_kl_from_logprobs, compute_token_logprobs_student, 
    compute_avg_exit_layer, extract_solution
)
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from shared_utils.load import get_tokenizer, configs_from_yaml


class TestEarlyExitModelConsistency(unittest.TestCase):
    """
    Unit test to validate that free generation outputs match student model's 
    top token predictions in greedy sampling mode.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up models and configurations once for all tests."""
        cls.num_test_examples = 3  # Number of examples to test
        cls.rl_hparams = RLHyperparams(k=3)
    
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        cls.config_path = "config_deepseek.yaml"
        cls.sft_model_path = "models/trained_model_v0"
        cls.artifact_path = 'vkarthik095-university-of-amsterdam/early-exit/early-exit-model-fs5ofmzp:v0'
        # Disable automatic differentiation for inference
        torch.set_grad_enabled(False)
        
        # Initialize tokenizer and config
        cls.tokenizer = get_tokenizer(cls.model_name)
        cls.config = configs_from_yaml(cls.config_path, cls.tokenizer.eos_token_id)
        
        cls.config['lora']['r'] = 8 
        cls.config['lora']['lora_alpha'] = 16
        
        cls.config['generation']['do_sample'] = False
        cls.config['generation']['temperature'] = None
        cls.config['generation']['top_k'] = None
        cls.config['generation']['top_p'] = None
        # Initialize student model
        cls.student = get_model(cls.model_name, cls.config['model'], cls.device)
        cls.student = replace_attention_layers(cls.student, cls.config['lora'], cls.device)
        cls.student = load_model_from_wandb(cls.student, cls.sft_model_path, cls.artifact_path)
        
        # Load dataset
        cls.dataset = load_dataset("gsm8k", "main").shuffle()
        
    def generate_mismatch_dataframe(self, prompt: str) -> Optional[pd.DataFrame]:
        """
        Generate completions and check for mismatches between free generation and student predictions.
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            DataFrame containing mismatches if any, None otherwise
        """
        # Generate K completions
        completions, exit_info = generate_k_completions(
            self.student, 
            [prompt], 
            k=self.rl_hparams.k,
            tokenizer=self.tokenizer, 
            config=self.config, 
            device=self.device,
            system_prompt=self.rl_hparams.system_prompt
        )
        # Get input prompt length
        input_prompt_length = get_input_prompt_length(
            self.tokenizer, 
            prompt, 
            system_prompt=self.rl_hparams.system_prompt
        )
        
        # Set student mode
        set_transformer_early_exit_mode(self.student, 'sft_student')
        
        
        # Prepare prescribed exit layers
        prescribed_exit_layers = pad_sequence(
            exit_info['prescribed_exit_layers'], 
            batch_first=True, 
            padding_value=torch.inf
        )
        
        # # Compute student log probabilities
        # stu_logprobs, student_early_exit_logprobs = compute_token_logprobs_student(
        #     self.student,
        #     completions['tokens'],
        #     prescribed_exit_layers=prescribed_exit_layers,
        #     input_prompt_length=input_prompt_length
        # )
        
        # Get student output scores
        student_output_scores, collected_exit_logits = self.student(
            completions['tokens'],
            prescribed_exit_layer_idxs=prescribed_exit_layers
        )
        
        # Calculate probabilities
        student_probs = student_output_scores.logits.softmax(-1)
        
        # Check for mismatches
        rows = []
        for indx in range(self.rl_hparams.k):
            start_idx = input_prompt_length - 1 # Using input_prompt_length instead of hardcoded 57
            end_idx = start_idx + len(exit_info['prescribed_exit_layers'][indx])
            
            # Check each token
            for idx in range(start_idx, end_idx):
                # Get student top prediction
                student_top_id = torch.argmax(student_probs[indx, idx]).item()
                student_top_prob = student_probs[indx, idx, student_top_id].item()
                student_top_token = self.tokenizer.decode([student_top_id])
                
                # Get actual generated token
                generated_token = self.tokenizer.decode([completions['tokens'][indx][idx + 1].item()])
                
                # Check for mismatch
                if student_top_token != generated_token:
                    exit_layer = exit_info['prescribed_exit_layers'][indx][idx - start_idx].item()
                    rows.append({
                        "Batch number": indx,
                        "Position": idx,
                        "Student Top Next Token": student_top_token,
                        "Student Prob": student_top_prob,
                        "Free Generated Next Token": generated_token,
                        "Free Generated Token Probs (student)": student_probs[indx, idx, completions['tokens'][indx][idx + 1]].item(),
                        "Exit layer": exit_layer if exit_layer != torch.inf else "inf",
                    })
        total_tokens = sum(len(layers) for layers in exit_info['prescribed_exit_layers'])
        if rows:
            return pd.DataFrame(rows), total_tokens
        return None, total_tokens

    def test_multiple_prompts_consistency(self):
        """Test consistency across multiple prompts from the dataset.
        
        Allows up to 1% token mismatch rate before failing.
        """
        # Verify greedy sampling is enabled
        self.assertFalse(
            self.config['generation'].get('do_sample', True),
            "Test requires greedy sampling (do_sample=False)"
        )
        num_test_examples = min(self.num_test_examples, len(self.dataset["train"]))
        all_mismatches = []
        total_tokens_all_examples = 0
        
        for i in range(num_test_examples):
            example = self.dataset["train"][i]
            prompt = example["question"]
            
            mismatch_df, num_tokens_this_example = self.generate_mismatch_dataframe(prompt)
            
            # Calculate tokens for this example
            total_tokens_all_examples += num_tokens_this_example
            
            if mismatch_df is not None:
                mismatch_df['Example_Index'] = i
                all_mismatches.append(mismatch_df)
        
        if all_mismatches:
            combined_df = pd.concat(all_mismatches, ignore_index=True)
            combined_df.to_csv("tests/test_multiple_mismatches.csv", index=False)
            
            mismatch_rate = len(combined_df) / total_tokens_all_examples
            
            # Only fail if mismatch rate > 1%
            if mismatch_rate > 0.01:
                error_msg = f"\nMismatch rate {mismatch_rate:.2%} exceeds 1% threshold\n"
                error_msg += f"Found mismatches in {len(all_mismatches)} out of {num_test_examples} examples\n"
                error_msg += f"Total mismatched tokens: {len(combined_df)} out of {total_tokens_all_examples}\n"
                error_msg += f"\n\nFull mismatch data saved to 'tests/test_multiple_mismatches.csv'"
                self.fail(error_msg)
            else:
                print(f"Acceptable mismatch rate: {mismatch_rate:.2%} ({len(combined_df)}/{total_tokens_all_examples} tokens)")


if __name__ == "__main__":
    # Run as unittest
    unittest.main()