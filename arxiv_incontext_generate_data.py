import time
import argparse
import json

import glob
import os
import ipdb
st = ipdb.set_trace

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

perplexity_model_name = "meta-llama/Llama-3.1-8B"
perplexity_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
perplexity_model = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
perplexity_device = 'cuda:1'
# perplexity_device = f'cuda:{args.perplexity_device}' if torch.cuda.is_available() and args.perplexity_device >= 0 else 'cpu'
perplexity_model.to(perplexity_device)

# Define command-line arguments
parser = argparse.ArgumentParser(description="SFT Trainer with extended configuration options")

# Training and evaluation arguments
parser.add_argument("--element_start", type=int, default=0)
parser.add_argument("--element_end", type=int, default=600)
parser.add_argument("--mode", type=str, default='train')

# Parse arguments
args = parser.parse_args()

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
    perplexity = -avg_log_prob


    return perplexity

generator2 = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=200,
    device=device
)

def remove_references_and_appendices(content, headings):
    for heading in headings:
        if 'References' in heading or 'references' in heading or 'REFERENCES' in heading or 'Bibliography' in heading or 'bibliography' in heading or 'BIBLIOGRAPHY' in heading:
            content = content[:content.find(heading)]
            return content
    return content

def get_abstract(content, headings):
    for idx, heading in enumerate(headings[:-1]):
        if 'Abstract' in heading or 'ABSTRACT' in heading:
            return content[content.find(heading):content.find(headings[idx+1])]
    return ''

def remove_abstract(content, headings):
    for idx, heading in enumerate(headings[:-1]):
        if 'Abstract' in heading or 'ABSTRACT' in heading:
            return content[content.find(headings[idx+1]):]
    return content

def remove_introduction(content, headings):
    for idx, heading in enumerate(headings[:-1]):
        if 'Introduction' in heading or 'INTRODUCTION' in heading:
            return content[content.find(headings[idx+1]):], idx+1
    return content, -1

def remove_related_work(content, headings):
    for idx, heading in enumerate(headings[:-1]):
        if 'Related Work' in heading or 'RELATED WORK' in heading or 'Related work' in heading:
            return content[content.find(headings[idx+1]):], idx+1
    return content, -1

def remove_all_but_three_sections(content, headings, start_idx):
    for idx, heading in enumerate(headings):
        if idx > start_idx + 2:
            content = content[:content.find(heading)]
            print(heading, headings, len(content))
            return content
    return content
        
def get_indices_of_equations(content):
    indices = []
    start_idx = 0
    while start_idx < len(content):

        eq_start_indices = set()
        eq_start_indices.add(content.find('$$', start_idx))
        eq_start_indices.add(content.find('\\begin{align}', start_idx))
        eq_start_indices.add(content.find('\\begin{equation}', start_idx))
        eq_start_indices.add(content.find('\\begin{align*}', start_idx))
        eq_start_indices.add(content.find('\\begin{equation*}', start_idx))

        if -1 in eq_start_indices:
            eq_start_indices.remove(-1)
        if len(eq_start_indices) == 0:
            break
        start_idx = min(eq_start_indices)

        if content.find('$$', start_idx) == start_idx:
            end_idx = content.find('$$', start_idx + 2)
            indices.append((start_idx, end_idx + 2))
            start_idx = end_idx + 2
        elif content.find('\\begin{align}', start_idx) == start_idx:
            end_idx = content.find('\\end{align}', start_idx)
            indices.append((start_idx, end_idx))
            start_idx = end_idx + 11
        elif content.find('\\begin{equation}', start_idx) == start_idx:
            end_idx = content.find('\\end{equation}', start_idx)
            indices.append((start_idx, end_idx))
            start_idx = end_idx + 14
        elif content.find('\\begin{align*}', start_idx) == start_idx:
            end_idx = content.find('\\end{align*}', start_idx)
            indices.append((start_idx, end_idx))
            start_idx = end_idx + 12
        elif content.find('\\begin{equation*}', start_idx) == start_idx:
            end_idx = content.find('\\end{equation*}', start_idx)
            indices.append((start_idx, end_idx))
            start_idx = end_idx + 15

    return indices    



all_md_files = glob.glob("/grogu/user/mprabhud/papers_o1/attention_papers/*/main.md")
all_headings = []
num_abstracts_found = 0

num_total_generations = 0
num_added_generations = 0

mode = args.mode

if not os.path.exists(f"/grogu/user/lilic/arxiv_cot/incontext_sft_{mode}"):
    os.makedirs(f"/grogu/user/lilic/arxiv_cot/incontext_sft_{mode}")

start_time = time.time()

for idx,md_file in enumerate(all_md_files):
    if idx < args.element_start or idx >= args.element_end:
        continue
    print(f"Processing paper {idx}, total time has been {time.time() - start_time} seconds")
    print(f'Num total generations: {num_total_generations}, Num added generations: {num_added_generations}')

    if os.path.exists(md_file):
        # print(f'{idx}/{len(all_md_files)}')
        with open(md_file, "r") as f:
            content = f.read()

        headings = []
        for line in content.split('\n')[1:]:
            if line.strip().startswith('#'):
                headings.append(line.strip())
            elif 'Abstract' in line:
                headings.append('Abstract')

        content = remove_references_and_appendices(content, headings)
        abstract = get_abstract(content, headings)
        content = remove_abstract(content, headings)
        content, start_idx_after_introduction = remove_introduction(content, headings)
        content, start_idx_after_related_work = remove_related_work(content, headings)
        start_idx = max(start_idx_after_introduction, start_idx_after_related_work)
        content = remove_all_but_three_sections(content, headings, start_idx)

        equation_indices = get_indices_of_equations(content)

        for _ in range(5):

            for eqn_start, eqn_end in equation_indices:
                messages = []
                prefix = abstract + content[:eqn_start]
                equation = content[eqn_start:eqn_end]
                prompt = f'Context: {prefix}\nEquation: {equation}\nInstruction: Given the context, please generate a short chain of thought (less than 3 sentences) to explain how you arrived at the equation.'

                messages.append({"role": "user", "content": prompt})

                generated_text = generator2(messages)[0]['generated_text'][-1]['content']
                print(generated_text)

                tokenized_prefix = perplexity_tokenizer(prefix, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_equation = perplexity_tokenizer(equation, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_model_output = perplexity_tokenizer(generated_text, return_tensors="pt").to(device)['input_ids'][0]

                old_perplexity = get_perplexity(tokenized_prefix, tokenized_equation)
                new_perplexity = get_perplexity(tokenized_prefix, tokenized_equation, tokenized_model_output)
                print(f"Old perplexity: {old_perplexity}, New perplexity: {new_perplexity}")

                num_total_generations += 1

                if new_perplexity < old_perplexity:
                    num_added_generations += 1
                    print('Adding to dataset')

                    new_item = {
                        'prefix': prefix,
                        'equation': equation,
                        'model_output': generated_text,
                        'old_perplexity': old_perplexity,
                        'new_perplexity': new_perplexity
                    }

                    consolidated_output_path = f"/grogu/user/lilic/arxiv_cot/incontext_sft_{mode}/elements_{args.element_start}_{args.element_end}.json"
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

        
        
print('done')
