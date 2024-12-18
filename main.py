import argparse
from trl import SFTConfig, SFTTrainer
from functools import partial
from transformers import Trainer
import hydra
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
from util import clean_numbers, last_boxed_only, last_boxed_only_string, remove_boxed
import re

DATA_DIR = os.environ['DATA_DIR']

if "CKPT_DIR" in os.environ:
    CKPT_DIR = os.environ['CKPT_DIR']
else:
    CKPT_DIR = None




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


def convert_item_incontext(item, rand_item=None, eos_token=None, cfg=None):
    messages = []
    
    prefix = item.get('prefix', '')
    sentence = item.get('sentence', '')
    model_output = item.get('model_output', '')
    old_perplexity = item.get('old_perplexity', '')
    new_perplexity = item.get('new_perplexity', '')

    if 'Sentence #' in model_output: # TODO fix this in dataset generation
        idx_of_next_sentence = model_output.find('Sentence #')
        model_output = model_output[:idx_of_next_sentence]
        
    if cfg.causal_llm:
        new_item = {
            'source': 'GPT4LLM',
            'prefix': prefix,
            'model_output': model_output,
            'sentence': sentence,
            'old_perplexity': old_perplexity,
            'new_perplexity': new_perplexity
        }
    else:
        if cfg.prompt_version == 1:
            prompt = prefix
        elif cfg.prompt_version == 2:
            prompt = f'Please complete the following text (in less than 200 tokens): {prefix}'
        elif cfg.prompt_version == 3:
            prompt = f'Please generate one sentence that completes the following text and do not generate any other text: {prefix}'
        else:
            raise NotImplementedError

        messages.append({'content': prompt, 'role': 'user'})
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


def convert_item_hendrycks_math(item, cfg=None):
    messages = []
    problem = item.get('problem', '')
    model_output = item.get('model_output', '') if 'model_output' in item else None
    solution = item.get('solution', '')
    
    if cfg.causal_llm:
        final_answer = last_boxed_only_string(solution)
        if final_answer is None:
            st()
        cot_idx = solution.find(final_answer)
        cot = solution[:cot_idx]
        new_item = {
            'source': 'GPT4LLM',
            'problem': problem,
            'cot': cot,
            'final_answer': final_answer,
        }        
        return new_item
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
    def __init__(self, test_dataset, tokenizer, generate_every_n_steps=100, num_samples=5, max_new_tokens=400, cfg=None):
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.generate_every_n_steps = generate_every_n_steps
        self.num_samples = num_samples
        self.accumulated_data = []
        self.max_new_tokens = max_new_tokens
        self.cfg = cfg


    def on_step_begin(self, args, state, control, **kwargs):
        with torch.no_grad():
            model = kwargs['model']
            tokenizer = self.tokenizer
            if state.global_step % self.generate_every_n_steps == 0:
                model.eval()
                generator = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    device=model.device,
                    do_sample=True,
                )
                
                
                if len(self.test_dataset) < self.num_samples:
                    sample_indices = np.arange(len(self.test_dataset))
                else:
                    sample_indices = np.arange(self.num_samples)
                samples = self.test_dataset.select(sample_indices)

                test_correct = 0
                test_total = 0

                for idx, sample in enumerate(samples):
                    if self.cfg.causal_llm:
                        prefix = sample['problem']
                        cot = sample['cot']
                        suffix = sample['final_answer']
                        input_text = "Question: " + prefix
                        
                        generated_text = generator(input_text)[0]['generated_text']
                        answer_idx = generated_text.find("Answer: ")
                        
                        if answer_idx != -1:
                            generated_answer = generated_text[answer_idx + len("Answer: "):]
                        else:
                            generated_answer = ''                        
                        
                        ground_truth_solution = suffix
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
                        ground_truth_solution = sample.get('solution', '')
                        input_text_vis = input_text[0]['content']
                    
                    # Extract answers and check correctness
                    boxed_answer = last_boxed_only_string(generated_text)
                    if boxed_answer is None:
                        generated_answer = ''
                    else:
                        generated_answer = remove_boxed(boxed_answer)
                    
                    ground_truth_answer = remove_boxed(last_boxed_only_string(ground_truth_solution))
                    # print(generated_answer, ground_truth_answer)

                    test_total += 1
                    if generated_answer == ground_truth_answer:
                        test_correct += 1

                    # Log individual samples for inspection
                    self.accumulated_data.append({
                        "global_step": state.global_step,
                        "input_text": input_text_vis,
                        'generated_answer': generated_answer,
                        "ground_truth_solution": ground_truth_solution,
                        "generated_output": generated_text
                    })

                # Calculate accuracy
                test_accuracy = test_correct / test_total if test_total > 0 else 0

                # Log generated samples and accuracies to wandb
                table = wandb.Table(columns=["global_step", "input_text", "generated_answer", "ground_truth_solution", "generated_output"])
                for data in self.accumulated_data:
                    table.add_data(
                        data["global_step"],
                        data["input_text"],
                        data["generated_answer"],
                        data["ground_truth_solution"],
                        data["generated_output"]
                    )
                wandb.log({
                    'Generated Samples Hendrycks Math': table,
                    'test_accuracy_hendrycks_math': test_accuracy,
                    'global_step': state.global_step
                })
                model.train()

def get_text_in_sentences(text):
    # Split on period, exclamation mark, question mark, or newline, followed by optional whitespace
    sentences = re.split(r'[.!?\n]\s*', text)
    # Filter out empty sentences and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence]
    numbered_sentences = [(f"Sentence #{i+1}: {sentence}") for i, sentence in enumerate(sentences)]
    return numbered_sentences, len(sentences)

class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, train_dataset, test_dataset, tokenizer, generate_every_n_steps=100, num_samples=5, max_new_tokens=400, perplexity_device='cpu', cfg=None, start_thinking_token=None, end_thinking_token=None, eos_token=None):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.generate_every_n_steps = generate_every_n_steps
        self.num_samples = num_samples
        self.accumulated_data = []
        self.max_new_tokens = max_new_tokens
        self.perplexity_device = perplexity_device
        self.causal_llm = cfg.causal_llm
        self.start_thinking_token = start_thinking_token
        self.end_thinking_token = end_thinking_token
        self.eos_token = eos_token
        self.cfg = cfg
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self.cfg.tune_only_lora:
            for name, param in kwargs['model'].named_parameters():
                if 'lora' not in name and param.requires_grad:
                    param.requires_grad = False            
        
        
    def on_step_begin(self, args, state, control, **kwargs):
        perplexity_device = self.perplexity_device
        with torch.no_grad():
            model = kwargs['model']
            tokenizer = self.tokenizer
            if state.global_step % self.generate_every_n_steps == 0 and (not self.cfg.skip_first_step or state.global_step > 0):
                # if optimizer and scheduler:
                #     optimizer_state = optimizer.state_dict()
                #     scheduler_state = scheduler.state_dict()
                #     original_scheduler_step = scheduler.step
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
                
                for idx, sample in enumerate(samples):
                    if self.causal_llm:
                        prefix = sample['prefix']
                        model_output = sample['model_output']
                        sentence = sample['sentence']
                        input_text = prefix
                        
                        
                        original_full_text = prefix  + self.start_thinking_token + model_output + self.end_thinking_token
                        original_full_text_wo_thinking = prefix
                        
                        gen_output = generator(input_text, return_tensors=True)
                        input_text_len = len(self.tokenizer(input_text)['input_ids'])
                        generated_text = self.tokenizer.decode(gen_output[0]['generated_token_ids'][input_text_len:], skip_special_tokens=False)
                        
                        start_thinking_idx = generated_text.find(self.start_thinking_token)
                        end_thinking_idx = generated_text.find(self.end_thinking_token)
                        
                        if start_thinking_idx != -1 and end_thinking_idx != -1:
                            generated_text_thinking = generated_text[start_thinking_idx:end_thinking_idx + len(self.end_thinking_token)]
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
                    sentence = sample['sentence'] + self.eos_token
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
                # if optimizer and scheduler:
                #     optimizer.load_state_dict(optimizer_state)
                #     scheduler.load_state_dict(scheduler_state)
                #     scheduler.step = original_scheduler_step

# Define PEFT config if enabled
def get_peft_config(cfg):
    if cfg.tune_special_tokens:
        modules_to_save = ["special_tokens", "lm_head_special"]
    else:
        modules_to_save = None
    if cfg.use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=cfg.lora_r, 
            lora_alpha=cfg.lora_alpha,
            modules_to_save=modules_to_save,
        )
        return peft_config
    return None

def formatting_prompts_func(examples, start_thinking_token, end_thinking_token, eos_token, cfg, tokenizer):
    if cfg.dataset_type == "sft":
        prefixes = examples["problem"]
        cots = examples["cot"]
        suffixes = examples["final_answer"]
    elif cfg.dataset_type == "pretrain":
        prefixes = examples["prefix"]
        cots = examples["model_output"]
        suffixes = examples["sentence"]
    texts = []
    for prefix, cot, suffix in zip(prefixes, cots, suffixes):
        if start_thinking_token is not None and end_thinking_token is not None:
            text = "Question: " + prefix + start_thinking_token + cot + end_thinking_token + "Answer: " + suffix + eos_token
        else:
            text = "Question: " + prefix + cot + "Answer: " + suffix + eos_token
        texts.append(str(text))
    return texts


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    # Parse arguments
    print("cfg", cfg)
    print("Ranks:", os.environ.get('LOCAL_RANK'))
    
    if cfg.load_exp is not None:
        load_base = "/".join(CKPT_DIR.split('/')[:-1])
        if cfg.running:
            load_dir = f"{load_base}/running/{cfg.load_exp}"
        else:
            load_dir = f"{load_base}/saved/{cfg.load_exp}"
        assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
        print("Loading from", load_dir)
    else:
        load_dir = None
    # st()
    # Initialize wandb
    if os.environ.get('LOCAL_RANK', '0') == '0':
        wandb.init(project="openwebmath-sft6", config=dict(cfg))


    # Load tokenizer and model
    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cfg=cfg)        
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    start_thinking_token = None
    end_thinking_token = None
    
    if cfg.use_special_tokens:
        if cfg.use_existing:
            start_thinking_token = '<|reserved_special_token_3|>'
            end_thinking_token = '<|reserved_special_token_4|>'
        else:
            start_thinking_token = '<start_thinking>'
            end_thinking_token = '<end_thinking>'
            tokenizer.add_special_tokens({'additional_special_tokens': [start_thinking_token, end_thinking_token]})
    
    model.resize_token_embeddings(len(tokenizer))
    
    
    if load_dir is not None:
        if cfg.resume:
            pass
        else:
            model = PeftModel.from_pretrained(model,load_dir)
            load_dir = None
    
    model.config.pad_token_id = tokenizer.pad_token_id
    eos_token = tokenizer.eos_token
    
    

    train_file = f"{DATA_DIR}/{cfg.dataset_name}/{cfg.data_version}/train.json"
    with open(train_file, 'r') as f:
        all_train_data = json.load(f)
    
    # Load test data
    test_file = f"{DATA_DIR}/{cfg.dataset_name}/{cfg.data_version}/test.json"
    with open(test_file, 'r') as f:
        all_test_data = json.load(f)  
    
    if cfg.dataset_type == "pretrain":
        # Load training data      
        train_data_transformed = [convert_item_incontext(item, eos_token=tokenizer.eos_token, cfg=cfg) for i, item in enumerate(all_train_data) if len(item['prefix']) > 5]
        test_data_transformed = [convert_item_incontext(item, eos_token=tokenizer.eos_token, cfg=cfg) for item in all_test_data if len(item['prefix']) > 5]    
    
    elif cfg.dataset_type == "sft":
        train_data_transformed = [convert_item_hendrycks_math(item, cfg=cfg) for item in all_train_data]
        test_data_transformed = [convert_item_hendrycks_math(item, cfg=cfg) for item in all_test_data]        
        


    if cfg.debug:
        train_dataset = Dataset.from_list(train_data_transformed[:1])
        test_dataset = train_dataset
    else:
        train_dataset = Dataset.from_list(train_data_transformed)
        test_dataset = Dataset.from_list(test_data_transformed)
    
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print("Number of test examples:", len(all_test_data))
        print("Number of training examples:", len(all_train_data))

    causal_llm = cfg.causal_llm
    
    output_dir = cfg.output_dir + '/' + (wandb.run.name if os.environ.get('LOCAL_RANK', '0') == '0' else 'tmp')
    if os.environ.get('LOCAL_RANK', '0') == '0':
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Configure training arguments
    training_args = SFTConfig(
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        output_dir=output_dir,
        push_to_hub=cfg.push_to_hub,
        ddp_find_unused_parameters=False,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        save_strategy=cfg.save_strategy,
        metric_for_best_model='eval_loss',
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        save_steps=cfg.save_steps,   
        eval_on_start=not cfg.skip_first_step,
        bf16=True,
        optim="adamw_torch_fused",  # Use fused optimizer
    )
    
    start_time = time.time()
    
    # Load Hendrycks Math test data
    # test_directory = f"{DATA_DIR}/MATH/MATH/test/prealgebra"
    # test_data = []
    # for filename in os.listdir(test_directory):
    #     if filename.endswith(".json"):
    #         file_path = os.path.join(test_directory, filename)
    #         with open(file_path, 'r') as f:
    #             test_data.append(json.load(f))
    # if os.environ.get('LOCAL_RANK', '0') == '0':
    #     print(f"Time taken to load Hendrycks Math test data: {time.time() - start_time} seconds")



    # Initialize callbacks
    callbacks = []
    if cfg.dataset_type == "sft":
        hendrycks_math_callback = HendrycksMathGenerateSamplesCallback(
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            generate_every_n_steps=cfg.generate_every_n_steps,
            num_samples=cfg.num_samples,
            max_new_tokens=cfg.max_new_tokens,
            cfg=cfg,
        )    
        callbacks.append(hendrycks_math_callback)
    
    if cfg.generate_samples:
        callback = GenerateSamplesCallback(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            generate_every_n_steps=cfg.generate_every_n_steps,
            num_samples=cfg.num_samples,
            max_new_tokens=cfg.max_new_tokens,
            cfg=cfg,
            start_thinking_token=start_thinking_token,
            end_thinking_token=end_thinking_token,
            eos_token=eos_token,
        )
        callbacks.append(callback)
    
    if causal_llm:
        formatting_prompts_func_input = partial(formatting_prompts_func, start_thinking_token=start_thinking_token, end_thinking_token=end_thinking_token, eos_token=eos_token, cfg=cfg, tokenizer=tokenizer)
    else:
        formatting_prompts_func_input = None

    trainer = SFTTrainer(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        formatting_func=formatting_prompts_func_input,
        peft_config=get_peft_config(cfg),
        args=training_args,
        callbacks=callbacks,
    )
    trainer.can_return_loss = True
    
    

    if os.environ.get('LOCAL_RANK', '0') == '0':
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameter_names = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad and 'lora' not in n]
        print("parameter_names without lora", parameter_names)
        print(f"Number of trainable parameters: {num_trainable_params:,}")


    
    # Train model
    trainer.train(resume_from_checkpoint=load_dir)

if __name__ == "__main__":
    main()