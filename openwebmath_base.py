import argparse
from trl import SFTConfig, SFTTrainer
from transformers import Trainer
from peft import PeftModel
import time
import ipdb
st = ipdb.set_trace
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
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
import re

DATA_DIR = os.environ['DATA_DIR']

if "CKPT_DIR" in os.environ:
    CKPT_DIR = os.environ['CKPT_DIR']
else:
    CKPT_DIR = None
# DATA_DIR = '/grogu/user/lilic'

# Define command-line arguments
parser = argparse.ArgumentParser(description="SFT Trainer with extended configuration options")

# Training and evaluation arguments
parser.add_argument("--causal_llm", action="store_true", help="Use causal LLM")
parser.add_argument("--learning_rate", type=float, default=2.0e-4, help="Learning rate for training")
parser.add_argument("--num_train_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--logging_steps", type=int, default=25, help="Logging interval in steps")
parser.add_argument("--save_steps", type=int, default=2500, help="Logging interval in steps")
parser.add_argument("--packing", action="store_true", help="Enable data packing")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
parser.add_argument("--output_dir", type=str, default=f"{CKPT_DIR}", help="Directory to save model outputs")
parser.add_argument("--load_dir", type=str, default=None, help="Directory to load model from")
parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy")
parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub after training")
parser.add_argument("--rand_train", action="store_true", help="Use random samples for training")
parser.add_argument("--use_n_shot_prompt", type=int, default=0, help="Whether to use n-shot prompt")
parser.add_argument("--max_new_tokens", type=int, default=400, help="Max new tokens to generate")
parser.add_argument("--num_samples", type=int, default=20, help="Samples for evaluation")
parser.add_argument("--generate_every_n_steps", type=int, default=500, help="Eval freq")
parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--prompt_version", type=int, default=1, help="Weight decay")
parser.add_argument('--perplexity_device', type=int, default=1, help='GPU device number for perplexity model (default: 1)')
parser.add_argument("--debug_with_single_example", action="store_true")
parser.add_argument("--eval_steps", type=int, default=500, help="Eval freq")

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


def convert_item_incontext(item, rand_item=None, eos_token=None, causal_llm=False):
    messages = []
    
    prefix = item.get('prefix', '')
    sentence = item.get('sentence', '')
    model_output = item.get('model_output', '')
    old_perplexity = item.get('old_perplexity', '')
    new_perplexity = item.get('new_perplexity', '')

    if 'Sentence #' in model_output: # TODO fix this in dataset generation
        assert False
        idx_of_next_sentence = model_output.find('Sentence #')
        model_output = model_output[:idx_of_next_sentence]
        

    if args.prompt_version == 1:
        prompt = prefix
    elif args.prompt_version == 2:
        prompt = f'Please complete the following text (in less than 200 tokens): {prefix}'
    elif args.prompt_version == 3:
        prompt = f'Please generate one sentence that completes the following text and do not generate any other text: {prefix}'
    else:
        raise NotImplementedError
    # if causal_llm:
    #     data = prefix + "<start_thinking>" + model_output + "<xot>" + sentence
    #     new_item = {"text": data}
    # else:
    messages.append({'content': prompt, 'role': 'user'})
    # messages.append({'content': model_output + sentence, 'role': 'assistant'})
    messages.append({'content': model_output + eos_token, 'role': 'assistant'})

    num_turns = len(messages)

    new_item = {
        'source': 'GPT4LLM',
        'messages': messages,
        'prefix': prefix,
        'model_output': model_output,
        'sentence': sentence,
        'old_perplexity': old_perplexity,
        'new_perplexity': new_perplexity
    }
    return new_item

# def convert_item_incontext(item, rand_item=None):
#     messages = []
#     text = item.get('text', '')
#     model_output_sentence = item.get('model_output_sentence', '')
#     model_output_cot = item.get('model_output_cot', '')
#     text_index = text.find(model_output_sentence[1:-1])
#     prefix = text[:text_index]
#     suffix = model_output_cot + text[text_index:]

#     messages.append({'content': prefix, 'role': 'user'})
#     messages.append({'content': suffix, 'role': 'assistant'})

#     num_turns = len(messages)

#     new_item = {
#         'source': 'GPT4LLM',
#         'messages': messages,
#     }
#     return new_item

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

def get_log_probs(tokens, model):
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

def get_perplexity(prefix, sentence, model, tokenizer):
    tokenized_prefix = tokenizer(prefix, return_tensors="pt")['input_ids'][0]
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")['input_ids'][0]
    # remove bos token from sentence
    tokenized_sentence = tokenized_sentence[1:]

    prefix_length = tokenized_prefix.shape[0]
    combined_tokens = torch.cat((tokenized_prefix, tokenized_sentence), dim=0).to(model.device)
    token_log_probs = get_log_probs(combined_tokens.unsqueeze(0), model)
    sentence_log_probs = token_log_probs[prefix_length-1:combined_tokens.shape[0]-1]
    avg_log_prob = sum(sentence_log_probs) / len(sentence_log_probs)
    # perplexity = math.exp(-avg_log_prob)
    perplexity = -avg_log_prob # TODO rename from perplexity to NLL

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
                    device=device
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

def get_text_in_sentences(text):
    # Split on period, exclamation mark, question mark, or newline, followed by optional whitespace
    sentences = re.split(r'[.!?\n]\s*', text)
    # Filter out empty sentences and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence]
    numbered_sentences = [(f"Sentence #{i+1}: {sentence}") for i, sentence in enumerate(sentences)]
    return numbered_sentences, len(sentences)

class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, train_dataset, test_dataset, tokenizer, generate_every_n_steps=100, num_samples=5, max_new_tokens=400, perplexity_device='cpu', causal_llm=False):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.generate_every_n_steps = generate_every_n_steps
        self.num_samples = num_samples
        self.accumulated_data = []
        self.max_new_tokens = max_new_tokens
        self.perplexity_device = perplexity_device
        self.causal_llm = causal_llm

    def on_step_begin(self, args, state, control, **kwargs):
        perplexity_device = self.perplexity_device
        # st()
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
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    device=model.device,
                    do_sample=True,
                )
                # st()
                
                # generator.model = PeftModel.from_pretrained(model = model)



                if len(self.train_dataset) == 1:
                    sample_indices = np.arange(1)
                    self.num_samples = 1
                else:
                    sample_indices = np.arange(0, self.num_samples * 10, 10)
                samples = concatenate_datasets([self.train_dataset.select(sample_indices), self.test_dataset.select(sample_indices)])

                gt_cot_perplexities_test = []
                gt_cot_perplexities_train = []
                generated_cot_perplexities_test = []
                generated_cot_perplexities_train = []
                gt_perplexities_test = []
                gt_perplexities_train = []

                print(model.base_model.model.model.embed_tokens.weight[-3:].sum())

                for idx, sample in enumerate(samples):
                    if self.causal_llm:
                        prefix = sample['prefix']
                        model_output = sample['model_output']
                        sentence = sample['sentence']
                        input_text = prefix
                        
                        original_full_text = prefix  + "<start_thinking>" + model_output + "<end_thinking>"
                        original_full_text_wo_thinking = prefix
                        
                        gen_output = generator(input_text, return_tensors=True)
                        input_text_len = len(self.tokenizer(input_text)['input_ids'])
                        generated_text = self.tokenizer.decode(gen_output[0]['generated_token_ids'][input_text_len:], skip_special_tokens=False)
                        
                        start_thinking_idx = generated_text.find('<start_thinking>')
                        end_thinking_idx = generated_text.find('<end_thinking>')
                        
                        if start_thinking_idx != -1 and end_thinking_idx != -1:
                            generated_text_thinking = generated_text[start_thinking_idx:end_thinking_idx + len('<end_thinking>')]
                        else:
                            generated_text_thinking = ''
                        
                        generated_full_text = prefix + generated_text_thinking
                        
                        
                        input_text_vis = input_text
                    else:
                        input_text = []
                        assistant_text = ''
                        for message in sample['messages']:
                            if message['role'] == 'user':
                                input_text.append(message)
                            elif message['role'] == 'assistant':
                                assistant_text += message['content']
                        generated_text = generator(input_text)[0]['generated_text'][-1]['content']
                        input_text_vis = input_text[0]['content']

                    prefix = sample['prefix']
                    sentence = sample['sentence'] + eos_token
                    model_output = sample['model_output']
                    
                    # print("\n\n", "model_output", model_output, "\n\n")
                    # print("\n\n", "generated_text", generated_text, "\n\n")
                    # print("\n\n", "sentence", sentence, "\n\n")
                    
                    generated_cot_perplexity = get_perplexity(generated_full_text, sentence, model, tokenizer)
                    gt_cot_perplexity = get_perplexity(original_full_text, sentence, model, tokenizer)
                    gt_perplexity = get_perplexity(original_full_text_wo_thinking, sentence, model, tokenizer)

                    if idx >= self.num_samples:
                        gt_cot_perplexities_test.append(gt_cot_perplexity)
                        generated_cot_perplexities_test.append(generated_cot_perplexity)
                        gt_perplexities_test.append(gt_perplexity)
                    else:
                        gt_cot_perplexities_train.append(gt_cot_perplexity)
                        generated_cot_perplexities_train.append(generated_cot_perplexity)
                        gt_perplexities_train.append(gt_perplexity)

                    # Log individual samples for inspection
                    self.accumulated_data.append({
                        "global_step": state.global_step,
                        "input_text": input_text_vis,
                        # "assistant": assistant_text,
                        "generated_output": generated_text,
                        "model_output": sample['model_output'],
                        "sentence": sample['sentence'],
                        "gt_cot_perplexity": gt_cot_perplexity,
                        "generated_cot_perplexity": generated_cot_perplexity,
                        "gt_perplexity": gt_perplexity,
                    })

                # Log generated samples and accuracies to wandb
                table = wandb.Table(columns=["global_step", "input_text", "generated_output", "sentence", "model_output", "gt_cot_perplexity", "generated_cot_perplexity", "gt_perplexity"])
                for data in self.accumulated_data:
                    table.add_data(
                        data["global_step"],
                        data["input_text"],
                        data["generated_output"],
                        data["sentence"],
                        data["model_output"],
                        data["gt_cot_perplexity"],
                        data["generated_cot_perplexity"],
                        data["gt_perplexity"],
                    )
                wandb.log({
                    'Generated Samples': table,
                    'global_step': state.global_step,
                    'gt_cot_perplexity_test': sum(gt_cot_perplexities_test) / len(gt_cot_perplexities_test),
                    'generated_cot_perplexity_test': sum(generated_cot_perplexities_test) / len(generated_cot_perplexities_test),
                    'gt_perplexity_test': sum(gt_perplexities_test) / len(gt_perplexities_test),
                    'gt_cot_perplexity_train': sum(gt_cot_perplexities_train) / len(gt_cot_perplexities_train),
                    'generated_cot_perplexity_train': sum(generated_cot_perplexities_train) / len(generated_cot_perplexities_train),
                    'gt_perplexity_train': sum(gt_perplexities_train) / len(gt_perplexities_train),
                })
                model.train()
                if optimizer and scheduler:
                    optimizer.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    scheduler.step = original_scheduler_step

# Define PEFT config if enabled
def get_peft_config(args):
    if args.use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,
            modules_to_save=["embed_tokens", "lm_head"],
        )
        return peft_config
    return None
# 'source': 'GPT4LLM',
# 'messages': messages,
# 'prefix': prefix,
# 'model_output': model_output,
# 'sentence': sentence,
# 'old_perplexity': old_perplexity,
# 'new_perplexity': new_perplexity
# }
def formatting_prompts_func(examples):
    global eos_token
    prefixes = examples["prefix"]
    model_outputs = examples["model_output"]
    sentences = examples["sentence"]
    texts = []
    for prefix, model_output, sentence in zip(prefixes, model_outputs, sentences):
        text = prefix + '<start_thinking>' + model_output + '<end_thinking>' + sentence + eos_token
        texts.append(str(text))
    return texts



if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    print(args)
    print("save steps", args.save_steps)
    print("Ranks:", os.environ.get('LOCAL_RANK'))

    # Initialize wandb
    if os.environ.get('LOCAL_RANK', '0') == '0':
        wandb.init(project="openwebmath-sft6", config=args)

    # Load training data
    if args.use_incontext:
        all_train_data = []
        train_dir = f"{DATA_DIR}/wikipedia_openwebmath/incontextv4_sft_train"
        for filename in os.listdir(train_dir):
            if filename.endswith('.json'):
                with open(os.path.join(train_dir, filename), 'r') as f:
                    data = json.load(f)
                    all_train_data.extend(data)
    else:
        with open(f"{DATA_DIR}/wikipedia_openwebmath/sft_train/chunk_0.json", 'r') as f:
            all_train_data = json.load(f)
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print(len(all_train_data))

    # Load tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        
    
    device = "cuda:0"
    model.to(device)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<start_thinking>','<end_thinking>']})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    eos_token = tokenizer.eos_token
    causal_llm = args.causal_llm

    if args.use_incontext:
        train_data_transformed = [convert_item_incontext(item, eos_token=tokenizer.eos_token, causal_llm=False) for i, item in enumerate(all_train_data) if len(item['prefix']) > 5]
    else:
        if args.rand_train:
            train_data_transformed = [convert_item(item, rand_item=all_train_data[np.random.randint(len(all_train_data))]) for i, item in enumerate(all_train_data)]
        else:
            train_data_transformed = [convert_item(item) for item in all_train_data]

    if args.debug_with_single_example:
        train_dataset = Dataset.from_list(train_data_transformed[:1])
    else:
        train_dataset = Dataset.from_list(train_data_transformed)

    # Load test data
    with open(f"{DATA_DIR}/wikipedia_openwebmath/incontextv4_sft_test/chunk_0_elements_0_50000_filtered.json", 'r') as f:
        all_test_data = json.load(f)
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print(len(all_test_data))

    if args.use_incontext:
        test_data_transformed = [convert_item_incontext(item, eos_token=tokenizer.eos_token) for item in all_test_data if len(item['prefix']) > 5]
    else:
        test_data_transformed = [convert_item(item) for item in all_test_data]
    test_dataset = Dataset.from_list(test_data_transformed)

    test_ds_privileged = load_from_disk(f"{DATA_DIR}/wikipedia_openwebmath/test/chunk_0")
    output_dir = args.output_dir + '/' + (wandb.run.name if os.environ.get('LOCAL_RANK', '0') == '0' else 'tmp')
    if os.environ.get('LOCAL_RANK', '0') == '0':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Define a custom SFTConfig class to override n_gpu
    class MySFTConfig(SFTConfig): # to prevent it from doing dataparallel even though multiple gpus are available
        @property
        def n_gpu(self):
            return 1

    # Configure training arguments
    training_args = MySFTConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        output_dir=output_dir,
        push_to_hub=args.push_to_hub,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        save_strategy='steps',
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        save_steps=args.save_steps,   
        bf16=True,
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
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print(f"Time taken to load Hendrycks Math test data: {time.time() - start_time} seconds")

    # Initialize callbacks
    callback = GenerateSamplesCallback(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        generate_every_n_steps=args.generate_every_n_steps,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        causal_llm=causal_llm
    )

    if causal_llm:
        formatting_prompts_func_input = formatting_prompts_func
    else:
        formatting_prompts_func_input = None
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        formatting_func=formatting_prompts_func_input,
        peft_config=get_peft_config(args),
        args=training_args,
        packing=False,
        callbacks=[callback] if os.environ.get('LOCAL_RANK', '0') == '0' else [],
    )
    trainer.can_return_loss = True
    model.can_return_loss = True
    
    if os.environ.get('LOCAL_RANK', '0') == '0':
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate number of trainable parameters
        parameter_names = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad and 'lora' not in n]
        print("parameter_names without lora", parameter_names)
        
        print(f"Number of trainable parameters: {num_trainable_params:,}")

    # Train model
    trainer.train(resume_from_checkpoint=args.load_dir)