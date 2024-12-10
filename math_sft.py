import argparse
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset, concatenate_datasets
import wandb
import json
import numpy as np
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import torch

# Define command-line arguments
parser = argparse.ArgumentParser(description="SFT Trainer with extended configuration options")

# Training and evaluation arguments
parser.add_argument("--learning_rate", type=float, default=2.0e-4, help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--logging_steps", type=int, default=25, help="Logging interval in steps")
parser.add_argument("--save_steps", type=int, default=500, help="Logging interval in steps")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
parser.add_argument("--output_dir", type=str, default="/grogu/user/lilic/hendrycks-sft", help="Directory to save model outputs")
parser.add_argument("--exp_id", type=str, default='0000', help="Experiment ID")

# PEFT arguments
parser.add_argument("--use_peft", action="store_true", help="Enable PEFT")
parser.add_argument("--lora_r", type=int, default=32, help="PEFT LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="PEFT LoRA alpha")

# Parse arguments
args = parser.parse_args()

# Initialize wandb
wandb.init(project="math-sft4", group=args.exp_id)

from util import clean_numbers, last_boxed_only, last_boxed_only_string
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def convert_item(item):
    messages = []
    problem = item.get('problem', '')
    model_output = item.get('model_output', '') if 'model_output' in item else None
    solution = item.get('solution', '')

    messages.append({
        'content': 'Problem: ' + problem + '\nPlease put your final answer in \\boxed{}.\nAnswer: ',
        'role': 'user'
    })

    if model_output is not None:
        messages.append({'content': model_output, 'role': 'assistant'})

    num_turns = len(messages)

    new_item = {
        'source': 'GPT4LLM',
        'messages': messages,
        'num_turns': num_turns,
        'solution': solution
    }
    return new_item

categories = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
all_data = []
for category in categories:
    if os.path.exists(f"/grogu/user/lilic/MATH/MATH/sftv2_train/{category}.json"):
        with open(f"/grogu/user/lilic/MATH/MATH/sftv2_train/{category}.json", 'r') as f:
            data = json.load(f)
            all_data.extend(data)

data_transformed = [convert_item(item) for item in all_data]
train_dataset = Dataset.from_list(data_transformed)

print(len(train_dataset))

# Load tokenizer and model
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
import torch
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
tokenizer.pad_token = tokenizer.eos_token

output_dir = args.output_dir + '/' + args.exp_id
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure training arguments
training_args = SFTConfig(
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    logging_steps=args.logging_steps,
    output_dir=output_dir,
    push_to_hub=False,
    packing=False,
    save_strategy='steps',
    save_steps=args.save_steps,
    bf16=True,
)

# Define PEFT config if enabled
def get_peft_config(args):
    if args.use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha
        )
        return peft_config
    return None

# Initialize the trainer with model and PEFT config
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    peft_config=get_peft_config(args),
    args=training_args,
)

# Train the model
trainer.train()
