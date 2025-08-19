import itertools
import torch
from tests.early_exit_teacher.early_exit_predictions import load_default_model_and_tokenizer, format_and_tokenize_input,\
                                    get_early_exit_indices, EarlyExitGenerator, PredictionObject,\
                                    KLExitGenerator
import numpy as np

from tests.early_exit_teacher.visualization import generate_multi_prompt_html_visualization, load_results_json, save_multi_prompt_results_html, save_results_json
# Example usage
if __name__ == "__main__":
    try:
        # Try to load existing results
        all_results = load_results_json("tests/early_exit_teacher/visualizations/early_exit_analysis.json")
        print("Using loaded results, skipping generation...")
    except FileNotFoundError:
        print("No existing results found, generating new predictions...")
        model, tokenizer = load_default_model_and_tokenizer(model_config_path = "config_deepseek.yaml")

        system_prompt = "You are a helpful programming tutor."
        prefiller = ""
        
        test_prompts = [
            "Explain the concept of recursion in programming.",
            "What are the key differences between supervised and unsupervised learning?",
            "Describe how HTTP works and what happens when you visit a website."
        ]
        
        max_new_tokens = 400
        all_modes = ['normal', 'unfrozen', 'frozen_residual', 'frozen_cache']
        early_exit_modes = [mode for mode in all_modes if mode != 'normal']
        kl_factors = [0.25, 0.5, 1.0, 4.0]
        # test_prompts = test_prompts[:1]; early_exit_modes = early_exit_modes[:1]; kl_factors = kl_factors[:1]   
        early_exit_combinations = list(itertools.product(early_exit_modes, kl_factors))
        all_combinations = [('normal', None)] + early_exit_combinations
        
        # Collect results for ALL prompts
        all_results = []
        
        for prompt_idx, prompt in enumerate(test_prompts):
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx + 1}: {prompt[:50]}...")
            print(f"{'='*60}")
            
            inputs = format_and_tokenize_input(
                prompt=prompt, 
                system_prompt=system_prompt,
                prefiller=prefiller,
                tokenizer=tokenizer, 
                device=model.device
            )
            
            # Collect results for this prompt
            prompt_results = []
            
            for mode, kl_factor in all_combinations:
                print(f"\n--- Mode: {mode}, KL Factor: {kl_factor} ---")
                
                # Initialize the generator
                kl_generator = KLExitGenerator(
                    model=model,
                    tokenizer=tokenizer,
                    mode=mode,
                    exitable_layers=get_early_exit_indices(model)
                )
                
                # Generate with KL-based early exit
                generated_tokens, chosen_layers, prediction = kl_generator.generate(
                    inputs=inputs,
                    kl_strength=kl_factor,
                    max_new_tokens=max_new_tokens
                )
                
                # Evaluate the response
                ans = kl_generator.evaluate_response(generated_tokens, chosen_layers, inputs)
                
                # Store results for this prompt
                prompt_results.append({
                    'mode': mode,
                    'kl_factor': kl_factor,
                    'response': tokenizer.decode(generated_tokens, skip_special_tokens=True),
                    'evaluation': ans
                })
            
            # Add this prompt's results to the overall collection
            all_results.append({
                'prompt': prompt,
                'results': prompt_results
            })
        save_results_json(all_results, filename="tests/early_exit_teacher/visualizations/early_exit_analysis.json")
        print("Results saved to JSON file.")
        
    output_file = "tests/early_exit_teacher/visualizations/early_exit_analysis.html"
    print(f"\n🎉 Saving results to: {output_file}")
    # Generate SINGLE HTML file with all prompts
    save_multi_prompt_results_html(all_results, filename=output_file)

        



