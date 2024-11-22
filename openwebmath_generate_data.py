from datasets import load_dataset, load_from_disk, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import matplotlib.pyplot as plt
import os
import time
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description="SFT Trainer with extended configuration options")

# Training and evaluation arguments
parser.add_argument("--chunk_start", type=float, default=0)
parser.add_argument("--chunk_end", type=float, default=1)
parser.add_argument("--mode", type=str, default='train')

# Parse arguments
args = parser.parse_args()

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

def get_perplexity(tokenized_prefix, tokenized_sentence, tokenized_model_output=None):
    if tokenized_model_output is not None:
        step_one_tokenized = tokenizer('\nStep 1: ', return_tensors="pt").to(device)['input_ids'][0]
        tokenized_prefix = torch.cat((tokenized_prefix, step_one_tokenized), dim=0).to(device)
        tokenized_prefix = torch.cat((tokenized_prefix, tokenized_model_output), dim=0).to(device)
    combined_tokens = torch.cat((tokenized_prefix, tokenized_sentence), dim=0).to(device)
    prefix_length = tokenized_prefix.shape[0]
    token_log_probs = get_log_probs(combined_tokens.unsqueeze(0))
    sentence_log_probs = token_log_probs[prefix_length-1:combined_tokens.shape[0]-1]
    avg_log_prob = sum(sentence_log_probs) / len(sentence_log_probs)
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def call_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    token_log_probs = get_log_probs(input_ids)
    input_tokens = input_ids[0]
    
    current_sentence = []
    current_sentence_log_prob = 0.0
    current_sentence_indices = []
    sentence_perplexity_pairs = []
    
    # Start from second token since we only have predictions for tokens after the first
    for i, token_id in enumerate(input_tokens[1:], 1):
        token = tokenizer.decode([token_id])
        current_sentence.append(token)
        current_sentence_log_prob += token_log_probs[i-1]  # i-1 because token_log_probs starts from first prediction
        current_sentence_indices.append(i)  # Count backwards from the end
        
        # Check if token ends with sentence boundary markers
        if token.endswith('\n') or any(token.rstrip().endswith(p) for p in ['.', '!', '?', '\n']):
            sentence = ''.join(current_sentence)#.strip()
            if sentence:  # Only add non-empty sentences
                # Convert log prob to perplexity
                sentence_perplexity = math.exp(-current_sentence_log_prob / len(current_sentence))
                sentence_perplexity_pairs.append((sentence, sentence_perplexity, current_sentence_indices))
            current_sentence = []
            current_sentence_log_prob = 0.0
            current_sentence_indices = []
    
    # Add the last sentence if exists
    if current_sentence:
        sentence = ''.join(current_sentence)#.strip()
        if sentence:
            sentence_perplexity = math.exp(-current_sentence_log_prob / len(current_sentence))
            sentence_perplexity_pairs.append((sentence, sentence_perplexity, current_sentence_indices))
    
    return input_tokens, sentence_perplexity_pairs


generate_prompt = """
{prefix}\n[Step 1: [FILL IN]\nStep 2: [FILL IN]\nStep 3: [FILL IN]]\n{sentence} {suffix}\n
Instruction: Please generate a step-by-step chain of thought to explain the sentence after. There should be exactly three (3) steps.
Answer: Step 1: 
"""

mode = args.mode

for chunk_num in range(int(args.chunk_start), int(args.chunk_end)):

    train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_{chunk_num}")

    start_time = time.time()

    if not os.path.exists(f"/grogu/user/lilic/filtered_openwebmath/sft_{mode}"):
        os.makedirs(f"/grogu/user/lilic/filtered_openwebmath/sft_{mode}")

    for i, item in enumerate(train_ds):
        if i % 10 == 0:
            print(f"Processed {i} items in {time.time() - start_time} seconds")

        metadata = json.loads(item["metadata"])
        math_score = metadata["extraction_info"]["math_score"]
        input_tokens, sentence_perplexity_pairs = call_model(item["text"])

        sorted_pairs = sorted(sentence_perplexity_pairs[1:], key=lambda x: x[1], reverse=True)
        max_perplexity_tuple = max(sorted_pairs, key=lambda x: x[1])
        split_idx = max_perplexity_tuple[2][0]  # Get the first index of the highest perplexity sentence

        prefix = input_tokens[:split_idx]
        suffix = input_tokens[split_idx + len(max_perplexity_tuple[2]):]  # Skip all indices of the highest perplexity sentence
        sentence = input_tokens[split_idx:split_idx + len(max_perplexity_tuple[2])]
        
        decoded_prefix = tokenizer.decode(prefix)
        decoded_suffix = tokenizer.decode(suffix)
        decoded_sentence = tokenizer.decode(sentence)

        prompt = generate_prompt.format(prefix=decoded_prefix, sentence=decoded_sentence, suffix=decoded_suffix)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(inputs["input_ids"].shape)
        if inputs["input_ids"].shape[1] > 1024:
            continue

        outputs = []
        for _ in range(2):
            outputs.append(model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + 200,  # Adjust as needed
                temperature=1.5,
                num_return_sequences=5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.9,
                top_k=50
            ))
        outputs = torch.cat(outputs, dim=0)

        # print(item["text"])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(prompt)
        # print('########################')

        # Iterate through all generated sequences
        num_saved = 0
        for j, output in enumerate(outputs):
            generated_tokens = output[inputs["input_ids"].shape[1]:]
            new_perplexity = get_perplexity(prefix, sentence, tokenized_model_output=generated_tokens)
            # old_perplexity = get_perplexity(prefix, sentence) # for some reason these are not exactly the same
            old_perplexity = max_perplexity_tuple[1]

            print(j, new_perplexity, old_perplexity, max_perplexity_tuple)
            # print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
            # print('--------------------------')

            if new_perplexity < old_perplexity:
                model_output = "\nStep 1: " + tokenizer.decode(generated_tokens, skip_special_tokens=True)
                num_saved += 1

                new_item = {
                    'prefix': decoded_prefix,
                    'model_output': model_output,
                    'sentence': decoded_sentence,
                    'suffix': decoded_suffix,
                    'old_perplexity': old_perplexity,
                    'new_perplexity': new_perplexity,
                    'output_index': j
                }

                consolidated_output_path = f"/grogu/user/lilic/filtered_openwebmath/sft_{mode}/chunk_{chunk_num}.json"
                if os.path.exists(consolidated_output_path):
                    with open(consolidated_output_path, 'r') as output_fp:
                        try:
                            all_data = json.load(output_fp)
                        except json.JSONDecodeError:
                            all_data = []  # Start with an empty list if the file is corrupted or empty
                else:
                    all_data = []  # Start with an empty list if the file does not exist

                all_data.append(new_item)

                # Write the updated list back to the single file
                with open(consolidated_output_path, 'w') as output_fp:
                    json.dump(all_data, output_fp, indent=4)

        print(f'Saved {num_saved} items')
