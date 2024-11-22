import argparse
from trl import SFTConfig, SFTTrainer
import time
import ipdb
st = ipdb.set_trace
from datasets import load_dataset, Dataset, concatenate_datasets
import wandb
import json
import numpy as np
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import torch
import math
import os 
from util import clean_numbers, last_boxed_only, last_boxed_only_string

DATA_DIR = os.environ['DATA_DIR']

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
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
parser.add_argument("--output_dir", type=str, default=f"{DATA_DIR}/hendrycks-sft-openwebmath", help="Directory to save model outputs")
parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub after training")
parser.add_argument("--rand_train", action="store_true", help="Use random samples for training")
parser.add_argument("--exp_id", type=str, default='0000', help="Experiment ID")
parser.add_argument("--use_n_shot_prompt", type=int, default=0, help="Whether to use n-shot prompt")
parser.add_argument("--max_new_tokens", type=int, default=400, help="Max new tokens to generate")
parser.add_argument("--num_samples", type=int, default=20, help="Samples for evaluation")
parser.add_argument("--generate_every_n_steps", type=int, default=500, help="Eval freq")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

# PEFT arguments
parser.add_argument("--use_incontext", action="store_true", help="incontext")
parser.add_argument("--use_peft", action="store_true", help="Enable PEFT")
parser.add_argument("--lora_r", type=int, default=32, help="PEFT LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=16, help="PEFT LoRA alpha")

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def convert_item(item, rand_item=None):
    messages = []
    prefix = item.get('prefix', '')
    model_output = item.get('model_output', '')
    sentence = item.get('sentence', '')
    suffix = item.get('suffix', '')

    messages.append({'content': prefix, 'role': 'user'})
    
    if rand_item is not None:
        rand_model_output = rand_item.get('model_output', '')
        messages.append({'content': rand_model_output + sentence, 'role': 'assistant'})
    else:
        messages.append({'content': model_output + sentence, 'role': 'assistant'})

    num_turns = len(messages)

    new_item = {
        'source': 'GPT4LLM',
        'messages': messages,
        'num_turns': num_turns,
        'sentence': sentence,
        'suffix': suffix
    }
    return new_item


def convert_item_incontext(item, rand_item=None):
    messages = []
    text = item.get('text', '')
    model_output_sentence = item.get('model_output_sentence', '')
    model_output_cot = item.get('model_output_cot', '')
    text_index = text.find(model_output_sentence[1:-1])
    prefix = text[:text_index]
    suffix = model_output_cot + text[text_index:]

    messages.append({'content': prefix, 'role': 'user'})
    messages.append({'content': suffix, 'role': 'assistant'})

    num_turns = len(messages)

    new_item = {
        'source': 'GPT4LLM',
        'messages': messages,
    }
    return new_item

def convert_item_hendrycks_math(item):
    messages = []
    problem = item.get('problem', '')
    model_output = item.get('model_output', '') if 'model_output' in item else None
    solution = item.get('solution', '')

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

def get_log_probs(tokens):
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[:, :-1, :]  # Remove last position's prediction
        target_ids = tokens[:, 1:]  # Shift right by 1 to get next tokens
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log prob of each actual token that appeared
        token_log_probs = []
        for i in range(target_ids.shape[1]):
            token_id = target_ids[0, i]
            token_log_prob = log_probs[0, i, token_id].item()
            token_log_probs.append(token_log_prob)
            
    return token_log_probs

def get_val_loss(tokenized_prefix, tokenized_sentence, tokenized_model_output=None):
    if tokenized_model_output is not None:
        tokenized_sentence = torch.cat((tokenized_model_output, tokenized_sentence), dim=0).to('cuda')
    prefix_length = tokenized_prefix.shape[0]
    combined_tokens = torch.cat((tokenized_prefix, tokenized_sentence), dim=0).to('cuda')
    token_log_probs = get_log_probs(combined_tokens.unsqueeze(0))
    sentence_log_probs = token_log_probs[prefix_length-1:combined_tokens.shape[0]-1]
    avg_log_prob = sum(sentence_log_probs) / len(sentence_log_probs)
    return avg_log_prob

def get_perplexity(tokenized_prefix, tokenized_sentence, tokenized_model_output=None):
    if tokenized_model_output is not None:
        tokenized_prefix = torch.cat((tokenized_prefix, tokenized_model_output), dim=0).to('cuda')
    prefix_and_model_output_length = tokenized_prefix.shape[0]
    combined_tokens = torch.cat((tokenized_prefix, tokenized_sentence), dim=0).to('cuda')
    token_log_probs = get_log_probs(combined_tokens.unsqueeze(0))
    sentence_log_probs = token_log_probs[prefix_and_model_output_length-1:combined_tokens.shape[0]-1]
    avg_log_prob = sum(sentence_log_probs) / len(sentence_log_probs)
    perplexity = math.exp(-avg_log_prob)
    return perplexity

class HendrycksMathGenerateSamplesCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, generate_every_n_steps=100, num_samples=5, max_new_tokens=400):
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
                samples = self.test_dataset.select(sample_indices)

                test_correct = 0
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
                    # print(generated_answer, ground_truth_answer)

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
                    'Generated Samples Hendrycks Math': table,
                    'test_accuracy_hendrycks_math': test_accuracy,
                    'global_step': state.global_step
                })
                model.train()
                if optimizer and scheduler:
                    optimizer.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    scheduler.step = original_scheduler_step

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

                # train_sample_indices = np.random.choice(len(self.train_dataset), self.num_samples, replace=False)
                # test_sample_indices = np.random.choice(len(self.test_dataset), self.num_samples, replace=False)
                # samples = concatenate_datasets([self.train_dataset.select(train_sample_indices), self.test_dataset.select(test_sample_indices)])
                sample_indices = np.arange(0, self.num_samples * 10, 10)
                samples = concatenate_datasets([self.train_dataset.select(sample_indices), self.test_dataset.select(sample_indices)])

                new_test_perplexities = []
                new_train_perplexities = []
                num_reduced_train_perplexities = 0
                num_reduced_test_perplexities = 0

                for idx, sample in enumerate(samples):
                    input_text = []
                    assistant_text = ''
                    for message in sample['messages']:
                        if message['role'] == 'user':
                            input_text.append(message)
                        elif message['role'] == 'assistant':
                            assistant_text += message['content']

                    generated_text = generator(input_text)[0]['generated_text'][-1]['content']

                    # Log individual samples for inspection
                    self.accumulated_data.append({
                        "global_step": state.global_step,
                        "input_text": input_text[0]['content'],
                        "assistant": assistant_text,
                        "generated_output": generated_text,
                        "sentence": sample['sentence'],
                        "suffix": sample['suffix']
                    })

                    tokenized_input = tokenizer(input_text[0]['content'], return_tensors="pt").to('cuda')['input_ids'][0]
                    tokenized_sentence = tokenizer(sample['sentence'], return_tensors="pt").to('cuda')['input_ids'][0]
                    tokenized_model_output = tokenizer(generated_text, return_tensors="pt").to('cuda')['input_ids'][0]
                    original_perplexity = get_perplexity(tokenized_input, tokenized_sentence, tokenized_model_output=None)
                    new_perplexity = get_perplexity(tokenized_input, tokenized_sentence, tokenized_model_output=tokenized_model_output)
                    if idx >= self.num_samples:
                        new_test_perplexities.append(new_perplexity)
                    else:
                        new_train_perplexities.append(new_perplexity)

                # Log generated samples and accuracies to wandb
                table = wandb.Table(columns=["global_step", "input_text", "assistant", "generated_output", "sentence", "suffix"])
                for data in self.accumulated_data:
                    table.add_data(
                        data["global_step"],
                        data["input_text"],
                        data["assistant"],
                        data["generated_output"],
                        data["sentence"],
                        data["suffix"]
                    )
                wandb.log({
                    'Generated Samples': table,
                    'global_step': state.global_step,
                    'new_test_perplexity': sum(new_test_perplexities) / len(new_test_perplexities),
                    'new_train_perplexity': sum(new_train_perplexities) / len(new_train_perplexities),
                })
                model.train()
                if optimizer and scheduler:
                    optimizer.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    scheduler.step = original_scheduler_step

# Define PEFT config if enabled
def get_peft_config(args):
    if args.use_peft:
        from trl import LoraConfig
        return LoraConfig(r=args.lora_r, alpha=args.lora_alpha)
    return None

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    print(args)

    # Initialize wandb
    wandb.init(project="openwebmath-sft", group=args.exp_id)

    # Load training data
    if args.use_incontext:
        all_train_data = []
        train_dir = "/home/mprabhud/datasets/o1/filtered_openwebmath/incontext_sft_train"
        for filename in os.listdir(train_dir):
            if filename.endswith('.json'):
                with open(os.path.join(train_dir, filename), 'r') as f:
                    data = json.load(f)
                    all_train_data.extend(data)
    else:
        with open(f"{DATA_DIR}/filtered_openwebmath/sft_train/chunk_0.json", 'r') as f:
            all_train_data = json.load(f)
    # print(len(all_train_data))
    if args.use_incontext:
        train_data_transformed = [convert_item_incontext(item) for i, item in enumerate(all_train_data)]
        # st()
    else:
        if args.rand_train:
            train_data_transformed = [convert_item(item, rand_item=all_train_data[np.random.randint(len(all_train_data))]) for i, item in enumerate(all_train_data)]
        else:
            train_data_transformed = [convert_item(item) for item in all_train_data]
    # st()
    train_dataset = Dataset.from_list(train_data_transformed)

    # Load test data
    with open(f"{DATA_DIR}/filtered_openwebmath/sft_test/chunk_0.json", 'r') as f:
        all_test_data = json.load(f)
    # print(len(all_test_data))

    test_data_transformed = [convert_item(item) for item in all_test_data]
    test_dataset = Dataset.from_list(test_data_transformed)

    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        push_to_hub=args.push_to_hub,
        packing=args.packing,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=50,                
        # save_strategy='steps',
        # save_steps=args.save_steps,
        bf16=True
    )
    
    start_time = time.time()
    
    # Load Hendrycks Math test data
    test_directory = f"{DATA_DIR}/MATH/MATH/test/prealgebra"
    test_data = []
    for filename in os.listdir(test_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(test_directory, filename)
            with open(file_path, 'r') as f:
                test_data.append(json.load(f))
    print(f"Time taken to load Hendrycks Math test data: {time.time() - start_time} seconds")
    
    test_data_transformed_hendrycks_math = [convert_item_hendrycks_math(item) for item in test_data]
    test_dataset_hendrycks_math = Dataset.from_list(test_data_transformed_hendrycks_math)

    # Initialize callbacks
    callback = GenerateSamplesCallback(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        generate_every_n_steps=args.generate_every_n_steps,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )

    hendrycks_math_callback = HendrycksMathGenerateSamplesCallback(
        test_dataset=test_dataset_hendrycks_math,
        tokenizer=tokenizer,
        generate_every_n_steps=args.generate_every_n_steps,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(args),
        args=training_args,
        callbacks=[hendrycks_math_callback],
    )
    
    # Calculate number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params:,}")
    # st()

    # Train model
    trainer.train()
