from datasets import load_dataset, load_from_disk, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import matplotlib.pyplot as plt
import os
import time
import argparse
import re
import transformers

# Define command-line arguments
parser = argparse.ArgumentParser(description="SFT Trainer with extended configuration options")

# Training and evaluation arguments
parser.add_argument("--chunk_start", type=int, default=0)
parser.add_argument("--chunk_end", type=int, default=1)
parser.add_argument("--element_start", type=int, default=0)
parser.add_argument("--element_end", type=int, default=1000)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--use_wikipedia", type=int, default=0)
parser.add_argument('--perplexity_device', type=int, default=1, help='GPU device number for perplexity model (default: 1)')

# Parse arguments
args = parser.parse_args()

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

perplexity_model_name = "meta-llama/Llama-3.1-8B"
perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
perplexity_device = f'cuda:{args.perplexity_device}' if torch.cuda.is_available() and args.perplexity_device >= 0 else 'cpu'
perplexity_model.to(perplexity_device)

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

def get_perplexity(tokenized_prefix, tokenized_sentence, tokenized_model_output=None):
    tokenized_prefix = tokenized_prefix.to(perplexity_device)
    tokenized_sentence = tokenized_sentence.to(perplexity_device)

    if tokenized_model_output is not None:
        tokenized_model_output = tokenized_model_output.to(perplexity_device)
        tokenized_prefix = torch.cat((tokenized_prefix, tokenized_model_output), dim=0).to(perplexity_device)
    prefix_and_model_output_length = tokenized_prefix.shape[0]
    combined_tokens = torch.cat((tokenized_prefix, tokenized_sentence), dim=0).to(perplexity_device)
    token_log_probs = get_log_probs(combined_tokens.unsqueeze(0), perplexity_model)
    sentence_log_probs = token_log_probs[prefix_and_model_output_length-1:combined_tokens.shape[0]-1]
    avg_log_prob = sum(sentence_log_probs) / len(sentence_log_probs)
    perplexity = math.exp(-avg_log_prob)


    return perplexity
    

def get_text_in_sentences(text):
    # Split on period, exclamation mark, question mark, or newline, followed by optional whitespace
    sentences = re.split(r'[.!?\n]\s*', text)
    # Filter out empty sentences and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence]
    numbered_sentences = [(f"Sentence #{i+1}: {sentence}") for i, sentence in enumerate(sentences)]
    return numbered_sentences, len(sentences)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=3,
    device='cuda'
)

generator2 = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    device='cuda'
)

mode = args.mode

for chunk_num in range(int(args.chunk_start), int(args.chunk_end)):
    start_time = time.time()
    
    if not args.use_wikipedia:
        train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/{mode}/chunk_{chunk_num}")
        if not os.path.exists(f"/grogu/user/lilic/filtered_openwebmath/incontextv3_sft_{mode}"):
            os.makedirs(f"/grogu/user/lilic/filtered_openwebmath/incontext_sftv3_{mode}")
    else:
        train_ds = load_from_disk(f"/grogu/user/lilic/wikipedia_openwebmath/{mode}/chunk_{chunk_num}")
        if not os.path.exists(f"/grogu/user/lilic/wikipedia_openwebmath/incontextv3_sft_{mode}"):
            os.makedirs(f"/grogu/user/lilic/wikipedia_openwebmath/incontextv3_sft_{mode}")


    num_total_generations = 0
    num_added_generations = 0

    for i, item in enumerate(train_ds):
        if i < args.element_start or i >= args.element_end:
            continue
        print(f"Processing element {i}, total time has been {time.time() - start_time} seconds")
        print(f'Num total generations: {num_total_generations}, Num added generations: {num_added_generations}')

        metadata = json.loads(item["metadata"])
        math_score = metadata["extraction_info"]["math_score"]

        lst_text_in_sentences, num_sentences = get_text_in_sentences(item["text"])
        text_in_sentences = " ".join(lst_text_in_sentences)

        if len(item["text"]) > 2000: continue

        with torch.no_grad():

            for j in range(5):
                messages = []
                messages.append({"role": "user", "content": f"{text_in_sentences}\nInstruction: Given the above text, please identify the sentence that is most difficult to understand. Please ignore anything that is not relevant, such as metadata. Please respond in the format of Sentence #<number>, where <number> is a number between 0 and {num_sentences}."})

                generated_text = generator(messages)[0]['generated_text'][-1]['content']

                index_of_hash = generated_text.find("#")
                if index_of_hash == -1:
                    print(f"No sentence number found in response: {generated_text}")
                    print('------------------')
                    continue
                sentence_num_str = generated_text[index_of_hash+1:].strip()
                if not sentence_num_str.isdigit():
                    print(f"Invalid sentence number format: {sentence_num_str}")
                    print('------------------')
                    continue
                sentence_num = int(sentence_num_str) - 1
                if sentence_num < 0 or sentence_num >= num_sentences:
                    print(f"Sentence number {sentence_num + 1} out of range (1-{num_sentences})")
                    print('------------------')
                    continue

                print(f"Number of sentences: {num_sentences}")
                print(lst_text_in_sentences[sentence_num])

                messages.append({"role": "assistant", "content": f"{generated_text}"})
                messages.append({"role": "user", "content": f"Please insert an additional sentence before the selected sentence that would make the sentence easier to understand. Please respond with Sentence #{sentence_num}.5: <sentence>. Do not generate any other text."})

                generated_text = generator2(messages)[0]['generated_text'][-1]['content']
                print(generated_text)

                processed_sentence = lst_text_in_sentences[sentence_num]
                idx_of_actual_sentence = processed_sentence.find(":")
                sentence = processed_sentence[idx_of_actual_sentence+1:].strip()
                location_of_sentence = item["text"].find(sentence)
                prefix = item["text"][:location_of_sentence]

                processed_model_output = generated_text
                idx_of_actual_model_output = processed_model_output.find(":")
                model_output = processed_model_output[idx_of_actual_model_output+1:].strip() + ' '

                tokenized_prefix = perplexity_tokenizer(prefix, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_sentence = perplexity_tokenizer(sentence, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_model_output = perplexity_tokenizer(model_output, return_tensors="pt").to(device)['input_ids'][0]

                old_perplexity = get_perplexity(tokenized_prefix, tokenized_sentence)
                new_perplexity = get_perplexity(tokenized_prefix, tokenized_sentence, tokenized_model_output)
                print(f"Old perplexity: {old_perplexity}, New perplexity: {new_perplexity}")

                num_total_generations += 1

                if new_perplexity < old_perplexity:
                    num_added_generations += 1
                    print('Adding to dataset')

                    new_item = {
                        'prefix': prefix,
                        'sentence': sentence,
                        'model_output': model_output,
                        'old_perplexity': old_perplexity,
                        'new_perplexity': new_perplexity
                    }

                    if not args.use_wikipedia:
                        consolidated_output_path = f"/grogu/user/lilic/filtered_openwebmath/incontextv3_sft_{mode}/chunk_{chunk_num}_elements_{args.element_start}_{args.element_end}.json"
                    else:
                        consolidated_output_path = f"/grogu/user/lilic/wikipedia_openwebmath/incontextv3_sft_{mode}/chunk_{chunk_num}_elements_{args.element_start}_{args.element_end}.json"
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
                print('------------------')
