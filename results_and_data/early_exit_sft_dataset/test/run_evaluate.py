import torch
import csv
import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader




import sys
sys.path.append("../../../")

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


#from shared_utils.data import CSVPromptDataset
from early_exit.util import get_model, CSVPromptDataset, save_model
from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text

from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode

import wandb
from datetime import datetime


# --- Configuration ---
# You can change this to "Qwen/Qwen2.5-3B-Instruct" or other sizes
MODEL_NAME = "Qwen/Qwen3-4B" 

INPUT_FILE = "rg_leg_counting_dataset.csv"
OUTPUT_FILE = "qwen_evaluation_results.csv"
model_config_path = "config_qwen3.yaml"                     # args.model_config_path

# System prompt exactly as requested
SYSTEM_PROMPT = (
    "I am going to give you a word problem listing various animals and their quantities. "
    "Your task is to calculate the total number of legs based on standard biological leg counts "
    "for healthy adult specimens (e.g., insects=6, arachnids=8, birds=2, mammals=4). "
    "If an animal typically has no legs (like a fish or snake), count it as 0.\n"
    "Output format:\n"
    "Reasoning: Breakdown the calculation for each animal type (count * legs = subtotal) and show the final summation.\n"
    "Answer: <integer only; no trailing punctuation>\n\n"
    "Example:\n"
    "Reasoning: 2 chickens have 2 legs each (2 * 2 = 4); 3 spiders have 8 legs each (3 * 8 = 24); "
    "1 snake has 0 legs (1 * 0 = 0). Total is 4 + 24 + 0 = 28.\n"
    "Answer: 28"
)

def get_model_and_tokenizer(model_name):
    print(f"Loading model: {model_name}...")
    
    # Auto-detects GPU (cuda), Mac (mps), or CPU
    device = "cuda" 
    print(f"Using device: {device}")

        
    # LOAD IN THE MODEL AND TOKENIZER
    tokenizer = get_tokenizer(model_name)
    config = configs_from_yaml(model_config_path, tokenizer.eos_token_id)
    model = get_model(model_name, config['model'], device)

    return model, tokenizer, device

def extract_answer(generated_text):
    """
    Parses the model output to find 'Answer: X'.
    Returns the answer string or None if not found.
    """
    # Look for the last occurrence of "Answer:" to avoid confusion with the example in the prompt
    if "Answer:" in generated_text:
        # Split by "Answer:" and take the last part
        part_after_answer = generated_text.split("Answer:")[-1]
        # Clean up: take the first line, strip whitespace/punctuation
        candidate = part_after_answer.strip().split("\n")[0].strip()
        # Remove potential trailing punctuation (like periods)
        return candidate.rstrip(".")
    return None

def main():
    # 1. Setup
    model, tokenizer, device = get_model_and_tokenizer(MODEL_NAME)
    
    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Make sure to run the generation script first.")
        return

    correct_count = 0
    total_count = 0

    # 3. Open Output CSV
    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "full_generation", "parsed_answer", "is_correct"])

        # 4. Evaluation Loop
        print("Starting evaluation...")
        for index, row in tqdm(df.iterrows(), total=len(df)):
            question = row["question"]
            expected_answer = str(row["answer"]).strip()

            # Format prompt using the chat template (Standard for Qwen/Llama)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            
            # Prepare inputs
            text_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text_prompt], return_tensors="pt").to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            # We strip the input tokens from the output to only get the new text
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse Answer
            model_answer = extract_answer(response_text)
            
            # Check correctness
            # We do a direct string compare, but you might want to try parsing to int if strictness varies
            if model_answer:
                is_correct = (model_answer == expected_answer)
            else:
                is_correct = False # Failed to follow format

            # Update stats
            total_count += 1
            if is_correct:
                correct_count += 1

            # Log to CSV
            writer.writerow([question, expected_answer, response_text, model_answer, is_correct])

    # 5. Final Report
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("-" * 30)
    print(f"Evaluation Complete.")
    print(f"Total Questions: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Detailed results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()