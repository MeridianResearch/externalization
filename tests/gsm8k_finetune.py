import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import wandb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import re
import pandas as pd
import json
import numpy as np

from shared_utils.load import get_tokenizer, configs_from_yaml
from shared_utils.generate import generate_text
from early_exit.util import get_model, load_model, save_model
from early_exit.patching import replace_attention_layers, set_transformer_early_exit_mode
from early_exit.rewards import extract_solution

device = "cuda"
base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
config_path = "config_deepseek.yaml"
sft_model_path = "models/early_exit_20250818_layers_5_kl1_0"
format_model_path = "models/gsm8k_trained"

learning_rate = 5e-5
num_epochs = 2
warmup_steps = 0
max_length = 512
gradient_accumulation_steps = 8
max_samples = 1000


def load_gsm8k_with_difficulty():
    gsm8k_dataset = load_dataset("gsm8k", "main")
    
    difficulty_dataset = load_dataset("lime-nlp/GSM8K_Difficulty", 'Difficulty Score')
    
    difficulty_df = pd.DataFrame(difficulty_dataset['train'])
    
    def categorize_difficulty(score):
        if score > 90:
            return "Easy"
        elif score >= 70:
            return "Medium"
        else:
            return "Hard"
    
    difficulty_df['difficulty_category'] = difficulty_df['solved_percentage'].apply(categorize_difficulty)
    difficulty_lookup = dict(zip(difficulty_df['problem'], difficulty_df[['solved_percentage', 'difficulty_category']].to_dict('records')))
    
    return gsm8k_dataset, difficulty_lookup

def create_format_examples(dataset_split, tokenizer, max_samples=None):
    gsm8k_dataset, difficulty_lookup = load_gsm8k_with_difficulty()
    data = gsm8k_dataset[dataset_split]
    
    examples = []
    for item in data:
        question = item["question"]
        full_solution = item["answer"]

        difficulty_info = difficulty_lookup.get(question, {'difficulty': None, 'difficulty_category': 'Unknown'})
        
        numerical_answer = extract_solution(full_solution, method="strict")
        if numerical_answer and difficulty_info['solved_percentage'] is not None:
            
            #extract the reasoning part before ####
            reasoning_match = re.search(r"(.*?)#### ", full_solution, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                continue  #skip examples without reasoning
            
            input_text = f"Question: {question}\n\nSolution:"
            
            target_text = f" {reasoning}\n\n#### {numerical_answer}"
            
            full_text = input_text + target_text
            encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt"
            )
            
            input_encoding = tokenizer(
                input_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt"
            )
            
            input_length = input_encoding["input_ids"].shape[1]
            
            labels = encoding["input_ids"].clone()
            labels[0, :input_length] = -100  #ignore loss when on input tokens
            
            examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
                "question": question,
                "target": target_text,
                "difficulty": difficulty_info['solved_percentage'],
                "difficulty_category": difficulty_info['difficulty_category']
            })
    
    if max_samples:
        easy_examples = [ex for ex in examples if ex['difficulty_category'] == 'Easy']
        medium_examples = [ex for ex in examples if ex['difficulty_category'] == 'Medium']
        hard_examples = [ex for ex in examples if ex['difficulty_category'] == 'Hard']
        
        samples_per_category = max_samples // 3
        remainder = max_samples % 3
        
        import random
        random.shuffle(easy_examples)
        random.shuffle(medium_examples)
        random.shuffle(hard_examples)
        
        selected_examples = []
        selected_examples.extend(easy_examples[:samples_per_category + (1 if remainder > 0 else 0)])
        selected_examples.extend(medium_examples[:samples_per_category + (1 if remainder > 1 else 0)])
        selected_examples.extend(hard_examples[:samples_per_category])
        
        random.shuffle(selected_examples)
        
        print(f"Selected {len(selected_examples)} examples:")
        print(f"  Easy: {len([ex for ex in selected_examples if ex['difficulty_category'] == 'Easy'])}")
        print(f"  Medium: {len([ex for ex in selected_examples if ex['difficulty_category'] == 'Medium'])}")
        print(f"  Hard: {len([ex for ex in selected_examples if ex['difficulty_category'] == 'Hard'])}")
        
        return selected_examples
    
    return all_examples

def collate_fn(batch):
    return batch[0]

def extract_assistant_response(full_response, input_text):
    response_text = full_response[len(input_text):].strip()
    
    assistant_marker = "<｜Assistant｜>"
    if assistant_marker in response_text:
        assistant_start = response_text.find(assistant_marker) + len(assistant_marker)
        response_text = response_text[assistant_start:].strip()
    
    end_markers = ["<｜end▁of▁sentence｜>", "<｜User｜>", "<｜System｜>"]
    for marker in end_markers:
        if marker in response_text:
            response_text = response_text.split(marker)[0].strip()
    
    return response_text

def log_evaluation_examples_to_wandb(eval_results_detailed, epoch, prefix="eval"):
    columns = ["epoch", "example_id", "difficulty", "question", "ground_truth", 
              "model_response", "extracted_answer", "correct_format", "correct_answer", 
              "reasoning_length"]
    
    table_data = []
    for i, example in enumerate(eval_results_detailed):
        table_data.append([
            epoch,
            i,
            example["difficulty_category"],
            example["question"],
            example["ground_truth"],
            example["response"],
            example["extracted_answer"],
            example["correct_format"],
            example["correct_answer"],
            example["reasoning_length"]
        ])
    
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({f"{prefix}_epoch_{epoch}": table})


def evaluate_accuracy(model, tokenizer, eval_examples, epoch=None, log_detailed=True):
    model.eval()
    
    test_data = eval_examples
    
    correct_format = 0
    correct_answer = 0
    total = len(test_data)

    math_system_prompt = """You are a helpful assistant that solves math word problems step by step. Always end your solution with '#### ' followed by the final number. For example, if the answer is 25, end with: #### 25"""
#     math_system_prompt = """You are a helpful assistant that solves math word problems step by step. Always end your solution with '#### ' followed by the final number.

# Here is an example:

# Question: Mary is 4 years old. Mike is 4 years older than Mary. How old is Anne if she is 1 year older than Mike?

# Solution: Mary is 4 years old. Since Mike is 4 years older than Mary's age of 4, Mike is age 4+4=8. If Anne is 1 year older than Mike's age of 8, then she is of age 8+1=9.

# #### 9"""

    category_stats = {
        'Easy': {'correct_format': 0, 'correct_answer': 0, 'total': 0, 'reasoning_lengths': []},
        'Medium': {'correct_format': 0, 'correct_answer': 0, 'total': 0, 'reasoning_lengths': []},
        'Hard': {'correct_format': 0, 'correct_answer': 0, 'total': 0, 'reasoning_lengths': []}
    }

    all_reasoning_lengths = []
    detailed_results = []
    
    for i, example in enumerate(test_data):
        question = example["question"]
        difficulty_category = example["difficulty_category"]

        ground_truth = extract_solution(example["target"], method="strict")
        
        input_text = f"Question: {question}\n\nSolution:"
        
        with torch.no_grad():
            response, exit_info = generate_text(
                model=model,
                prompt=input_text,
                system_prompt=math_system_prompt,
                prefiller="",
                tokenizer=tokenizer,
                generation_config={
                    'max_new_tokens': 400,
                    'do_sample': True,
                    #'temperature': 0.7,
                    #'top_p': 0.9,
                    'pad_token_id': tokenizer.eos_token_id
                },
                device=device
            )
        
        #get clean response
        response_text = extract_assistant_response(response, input_text)
        
        extracted_answer = extract_solution(response_text, method="strict")
        is_correct_format = extracted_answer is not None
        is_correct_answer = extracted_answer == ground_truth if extracted_answer else False

        #calc reasoning length before last ####
        reasoning_part = response_text
        if '####' in response_text:
            reasoning_part = response_text.rsplit('####', 1)[0].strip()  # rsplit with maxsplit=1 gets everything before last ####
        reasoning_length = len(reasoning_part.split())
        
        all_reasoning_lengths.append(reasoning_length)

        detailed_results.append({
            "question": question,
            "difficulty_category": difficulty_category,
            "ground_truth": str(ground_truth),
            "response": response_text,
            "extracted_answer": str(extracted_answer) if extracted_answer else "None",
            "correct_format": is_correct_format,
            "correct_answer": is_correct_answer,
            "reasoning_length": reasoning_length
        })
        
        #count metrics
        if is_correct_format:
            correct_format += 1
        if is_correct_answer:
            correct_answer += 1

        category_stats[difficulty_category]['total'] += 1
        category_stats[difficulty_category]['reasoning_lengths'].append(reasoning_length)
        if is_correct_format:
            category_stats[difficulty_category]['correct_format'] += 1
        if is_correct_answer:
            category_stats[difficulty_category]['correct_answer'] += 1
        
        if i < 6:
            print(f"\n--- Example {i+1} ---")
            print(f"Difficulty: {difficulty_category}")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Clean Response: {response_text}")
            print(f"Extracted Answer: {extracted_answer}")
            print(f"Has Correct Format: {is_correct_format}")
            print(f"Is Correct Answer: {is_correct_answer}")
    
    format_accuracy = correct_format / total
    answer_accuracy = correct_answer / total
    
    print(f"Format Accuracy: {format_accuracy:.3f} ({correct_format}/{total}) - Has #### format")
    print(f"Answer Accuracy: {answer_accuracy:.3f} ({correct_answer}/{total}) - Correct format + correct answer")
    
    results = {
        'format_accuracy': format_accuracy,
        'answer_accuracy': answer_accuracy,
        'avg_reasoning_length': np.mean(all_reasoning_lengths),
        'median_reasoning_length': np.median(all_reasoning_lengths)
    }

    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        if stats['total'] > 0:
            cat_format_acc = stats['correct_format'] / stats['total']
            cat_answer_acc = stats['correct_answer'] / stats['total']
            
            print(f"{category} Format Accuracy: {cat_format_acc:.3f} ({stats['correct_format']}/{stats['total']})")
            print(f"{category} Answer Accuracy: {cat_answer_acc:.3f} ({stats['correct_answer']}/{stats['total']})")
            
            results[f'{category}_format_accuracy'] = cat_format_acc
            results[f'{category}_answer_accuracy'] = cat_answer_acc
            if stats['reasoning_lengths']:
                results[f'{category.lower()}_avg_reasoning_length'] = np.mean(stats['reasoning_lengths'])
    
    if log_detailed and epoch is not None:
        log_evaluation_examples_to_wandb(detailed_results, epoch)
        
    model.train()
    return results


tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, sft_model_path)

set_transformer_early_exit_mode(model, 'free_generate')
#set_transformer_early_exit_mode(model, 'off')
model.train()
#model.enable_adapters()
model._early_exit_logs = []
model._early_exit_probabilities = []

train_examples = create_format_examples("train", tokenizer, max_samples=max_samples)
eval_examples = create_format_examples("test", tokenizer, max_samples=60)

train_loader = DataLoader(train_examples, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_examples, batch_size=1, shuffle=False)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=learning_rate
)

total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
#scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
scheduler = None

wandb.init(
    project="gsm8k-finetuning",
    config={
        "model_name": base_model_name,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "max_samples": max_samples,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
)

initial_results = evaluate_accuracy(model, tokenizer, eval_examples, epoch=0, log_detailed=True)
print(f"Initial format accuracy: {initial_results['format_accuracy']:.3f}")
print(f"Initial answer accuracy: {initial_results['answer_accuracy']:.3f}")
wandb.log({
    **initial_results,
    "epoch": 0,
    "step": 0
})

model.train()
global_step = 0

for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for step, batch in enumerate(progress_bar):
        if isinstance(batch, list):
            batch = batch[0]

        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        epoch_loss += loss.item()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            log_dict = {
                "loss": loss.item() * gradient_accumulation_steps,
                "step": global_step,
                "epoch": epoch + 1,
            }
            if scheduler:
                log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            else:
                log_dict["learning_rate"] = learning_rate
            
            wandb.log(log_dict)
        
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    results = evaluate_accuracy(model, tokenizer, eval_examples, epoch=epoch+1, log_detailed=True)
    print(f"Epoch {epoch+1} - Format accuracy: {results['format_accuracy']:.3f}")
    print(f"Epoch {epoch+1} - Answer accuracy: {results['answer_accuracy']:.3f}")
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": epoch_loss / len(train_loader),
        **results
    })

print(f"Saving format-tuned model to {format_model_path}")
save_model(model, format_model_path, upload_to_wandb=True)

#final_results = evaluate_accuracy(model, tokenizer, eval_examples)

print(f"Format improvement: {initial_results['format_accuracy']:.3f} to {results['format_accuracy']:.3f}")
print(f"Answer improvement: {initial_results['answer_accuracy']:.3f} to {results['answer_accuracy']:.3f}")

wandb.finish()