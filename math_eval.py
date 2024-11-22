
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
parser.add_argument("--packing", action="store_true", help="Enable data packing")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name")
parser.add_argument("--output_dir", type=str, default="hendrycks-sft", help="Directory to save model outputs")
parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub after training")
parser.add_argument("--exp_id", type=str, default='0000', help="Experiment ID")
parser.add_argument("--use_n_shot_prompt", type=int, default=0, help="Whether to use n-shot prompt")
parser.add_argument("--max_new_tokens", type=int, default=400, help="Max new tokens to generate")
parser.add_argument("--num_samples", type=int, default=20, help="Samples for evaluation")
parser.add_argument("--generate_every_n_steps", type=int, default=500, help="Eval freq")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

parser.add_argument("--checkpoint_num", type=int, default=500, help="Checkpoint to evaluate")

# PEFT arguments
parser.add_argument("--use_peft", action="store_true", help="Enable PEFT")
parser.add_argument("--lora_r", type=int, default=32, help="PEFT LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="PEFT LoRA alpha")

# Parse arguments
args = parser.parse_args()

# Initialize wandb
wandb.init(project="math-sft3-eval", group=args.exp_id)

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

    if args.use_n_shot_prompt:
        messages.append({'content': train_prompt + 'Problem: ' + problem + ' Please put your final answer in \\boxed{}.\nAnswer: ', 'role': 'user'})
    else:
        messages.append({'content': 'Problem: ' + problem + ' Please put your final answer in \\boxed{}.\nAnswer: ', 'role': 'user'})

    numerical_solution = last_boxed_only_string(solution)
    if model_output is not None:
        messages.append({'content': model_output + f'\nThe answer is {numerical_solution}' , 'role': 'assistant'})

    num_turns = len(messages)

    new_item = {
        'source': 'GPT4LLM',
        'messages': messages,
        'num_turns': num_turns,
        'solution': solution
    }
    return new_item


train_directory = "/grogu/user/lilic/MATH/MATH/train/prealgebra"

train_data = []

for filename in os.listdir(train_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(train_directory, filename)
        with open(file_path, 'r') as f:
            train_data.append(json.load(f))

train_data_transformed = [convert_item(item) for item in train_data]
train_dataset = Dataset.from_list(train_data_transformed)

test_directory = "/grogu/user/lilic/MATH/MATH/test/prealgebra"

test_data = []

for filename in os.listdir(test_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(test_directory, filename)
        with open(file_path, 'r') as f:
            test_data.append(json.load(f))

test_data_transformed = [convert_item(item) for item in test_data]
test_dataset = Dataset.from_list(test_data_transformed)

# Load tokenizer and model
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
# import torch
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
tokenizer.pad_token = tokenizer.eos_token

output_dir = 'hendrycks-sft/' + args.exp_id
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
    push_to_hub=args.push_to_hub,
    packing=args.packing,
    save_strategy='no',
    # bf16=True
)

# Define PEFT config if enabled
def get_peft_config(args):
    if args.use_peft:
        from trl import LoraConfig
        return LoraConfig(r=args.lora_r, alpha=args.lora_alpha)
    return None

class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, train_dataset, test_dataset, tokenizer, generate_every_n_steps=100, num_samples=5, max_new_tokens=400):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.generate_every_n_steps = generate_every_n_steps
        self.num_samples = num_samples
        self.accumulated_data = []
        self.max_new_tokens = max_new_tokens

    def on_step_begin(self, args, state, control, **kwargs):
        with torch.no_grad():
            model = kwargs['model']
            optimizer = kwargs.get('optimizer')
            scheduler = kwargs.get('lr_scheduler')
            print(optimizer)
            print(scheduler)
            tokenizer = self.tokenizer

            if state.global_step % self.generate_every_n_steps == 0:
                if optimizer and scheduler:
                    optimizer_state = optimizer.state_dict()
                    scheduler_state = scheduler.state_dict()
                    original_scheduler_step = scheduler.step
                model.eval()
                generator = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    device='cuda'
                )
                sample_indices = np.arange(self.num_samples)
                samples = concatenate_datasets([self.train_dataset.select(sample_indices), self.test_dataset.select(sample_indices)])
                # train_sample_indices = np.random.choice(len(self.train_dataset), self.num_samples, replace=False)
                # test_sample_indices = np.random.choice(len(self.test_dataset), self.num_samples, replace=False)
                # samples = concatenate_datasets([self.train_dataset.select(train_sample_indices), self.test_dataset.select(test_sample_indices)])

                train_correct = 0
                test_correct = 0
                train_total = 0
                test_total = 0

                for idx, sample in enumerate(samples):
                    input_text = []
                    assistant_text = ''
                    for message in sample['messages']:
                        if message['role'] == 'user':
                            input_text.append(message)
                        elif message['role'] == 'assistant':
                            assistant_text += message['content']

                    generated_text = generator(input_text)[0]['generated_text'][-1]['content']
                    ground_truth_solution = sample.get('solution', '')

                    # Extract answers and check correctness
                    generated_answer = remove_boxed(last_boxed_only_string(generated_text))
                    ground_truth_answer = remove_boxed(last_boxed_only_string(ground_truth_solution))
                    print(generated_answer, ground_truth_answer)

                    if idx < self.num_samples:  # train sample
                        train_total += 1
                        if generated_answer == ground_truth_answer:
                            train_correct += 1
                    else:  # test sample
                        test_total += 1
                        if generated_answer == ground_truth_answer:
                            test_correct += 1

                    # Log individual samples for inspection
                    self.accumulated_data.append({
                        "global_step": state.global_step,
                        "input_text": input_text[0]['content'],
                        "assistant": assistant_text,
                        "ground_truth_solution": ground_truth_solution,
                        "generated_output": generated_text
                    })

                # Calculate accuracy
                train_accuracy = train_correct / train_total if train_total > 0 else 0
                test_accuracy = test_correct / test_total if test_total > 0 else 0

                # Log generated samples and accuracies to wandb
                table = wandb.Table(columns=["global_step", "input_text", "assistant", "ground_truth_solution", "generated_output"])
                for data in self.accumulated_data:
                    table.add_data(
                        data["global_step"],
                        data["input_text"],
                        data["assistant"],
                        data["ground_truth_solution"],
                        data["generated_output"]
                    )
                wandb.log({
                    'Generated Samples': table,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'global_step': state.global_step
                })
                model.train()
                if optimizer and scheduler:
                    optimizer.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    scheduler.step = original_scheduler_step
                print(1/0)



# Initialize the callback
callback = GenerateSamplesCallback(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    tokenizer=tokenizer,
    generate_every_n_steps=1,  # Adjust as needed
    num_samples=args.num_samples,  # Adjust as needed
    max_new_tokens=args.max_new_tokens
)

# Initialize the trainer with model and PEFT config
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    peft_config=get_peft_config(args),
    args=training_args,
    callbacks=[callback],  # Include the callback here
)

if args.checkpoint_num == -1:
    trainer.train()
else:
    checkpoint_path = f"/grogu/user/lilic/hendrycks-sft/{args.exp_id}/checkpoint-{args.checkpoint_num}"
    trainer.train(resume_from_checkpoint=checkpoint_path)
