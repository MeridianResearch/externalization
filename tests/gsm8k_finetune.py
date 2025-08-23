import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import wandb
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import re
import pandas as pd

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
max_samples = 2000

#math_system_prompt = """You are a helpful assistant that solves math word problems step by step. Always end your solution with '#### <numerical_answer> where <numerical_answer> is just the final number."""


def create_format_examples(dataset_split, tokenizer, max_samples=None):
    dataset = load_dataset("gsm8k", "main")
    data = dataset[dataset_split]
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    examples = []
    for item in data:
        question = item["question"]
        full_solution = item["answer"]
        
        numerical_answer = extract_solution(full_solution, method="strict")
        if numerical_answer:
            
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
                "target": target_text
            })
    
    return examples

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


def evaluate_accuracy(model, tokenizer, eval_examples, num_samples=20):
    model.eval()
    
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"].select(range(min(num_samples, len(dataset["test"]))))
    
    correct_format = 0
    correct_answer = 0
    total = len(test_data)

    math_system_prompt = """You are a helpful assistant that solves math word problems step by step. Always end your solution with '#### ' followed by the final number. For example, if the answer is 25, end with: #### 25"""
    
    for i, example in enumerate(test_data):
        question = example["question"]
        # Extract ground truth from GSM8K format
        ground_truth = extract_solution(example["answer"], method="strict")
        
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
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'pad_token_id': tokenizer.eos_token_id
                },
                device=device
            )
        
        # Use the helper function to extract clean response
        response_text = extract_assistant_response(response, input_text)
        
        extracted_answer = extract_solution(response_text, method="strict")
        is_correct_format = extracted_answer is not None
        is_correct_answer = extracted_answer == ground_truth if extracted_answer else False
        
        # Count metrics
        if is_correct_format:
            correct_format += 1
        if is_correct_answer:
            correct_answer += 1
        
        if i < 3:
            print(f"\n--- Example {i+1} ---")
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
    
    model.train()
    return {
        'format_accuracy': format_accuracy,
        'answer_accuracy': answer_accuracy
    }


tokenizer = get_tokenizer(base_model_name)
config = configs_from_yaml(config_path, tokenizer.eos_token_id)

base_model = get_model(base_model_name, config['model'], device)
model = replace_attention_layers(base_model, config['lora'], device)
model = load_model(model, sft_model_path)

set_transformer_early_exit_mode(model, 'free_generate')
#set_transformer_early_exit_mode(model, 'sft_teacher')
model.train()
model._early_exit_logs = []

train_examples = create_format_examples("train", tokenizer, max_samples=max_samples)
eval_examples = create_format_examples("test", tokenizer, max_samples=100)

train_loader = DataLoader(train_examples, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_examples, batch_size=1, shuffle=False)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=learning_rate
)

total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

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

initial_results = evaluate_accuracy(model, tokenizer, eval_examples)
print(f"Initial format accuracy: {initial_results['format_accuracy']:.3f}")
print(f"Initial answer accuracy: {initial_results['answer_accuracy']:.3f}")
wandb.log({
    "format_accuracy": initial_results['format_accuracy'], 
    "answer_accuracy": initial_results['answer_accuracy'],
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
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            wandb.log({
                "loss": loss.item() * gradient_accumulation_steps,
                "learning_rate": scheduler.get_last_lr()[0],
                "step": global_step
            })
        
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    results = evaluate_accuracy(model, tokenizer, eval_examples)
    print(f"Epoch {epoch+1} - Format accuracy: {results['format_accuracy']:.3f}")
    print(f"Epoch {epoch+1} - Answer accuracy: {results['answer_accuracy']:.3f}")
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": epoch_loss / len(train_loader),
        "format_accuracy": results['format_accuracy'],
        "answer_accuracy": results['answer_accuracy']
    })

print(f"Saving format-tuned model to {format_model_path}")
save_model(model, format_model_path, upload_to_wandb=True)

final_results = evaluate_accuracy(model, tokenizer, eval_examples)
print(f"Final format accuracy: {final_results['format_accuracy']:.3f}")
print(f"Final answer accuracy: {final_results['answer_accuracy']:.3f}")
wandb.log({
    "final_format_accuracy": final_results['format_accuracy'],
    "final_answer_accuracy": final_results['answer_accuracy']
})
print(f"Format improvement: {initial_results['format_accuracy']:.3f} to {final_results['format_accuracy']:.3f}")
print(f"Answer improvement: {initial_results['answer_accuracy']:.3f} to {final_results['answer_accuracy']:.3f}")

wandb.finish()