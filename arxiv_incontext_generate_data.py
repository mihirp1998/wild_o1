import time
import argparse
import json

import glob
import os
import ipdb
st = ipdb.set_trace
from collections import defaultdict
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
            return content[:content.find(headings[idx])] + content[content.find(headings[idx+1]):]
    return content

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
# Sort files based on the numeric directory name
all_md_files.sort(key=lambda x: int(x.split('/')[-2]))
all_headings = []
num_abstracts_found = 0

num_total_generations = 0
num_added_generations = 0

mode = args.mode
datadir = os.environ['DATA_DIR']

if not os.path.exists(f"{datadir}/arxiv_cot/incontext_sft_{mode}"):
    os.makedirs(f"{datadir}/arxiv_cot/incontext_sft_{mode}")

start_time = time.time()

one_shot_prompt = """
Context: # Abstract

We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances in language modeling such as GPT- $\\mathbf{\\nabla}\\cdot\\mathbf{X}$ and BERT. In particular, we present Decision Transformer, an architecture that casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks.

# 2 Preliminaries

Transformers were proposed by Vaswani et al. [1] as an architecture to efficiently model sequences. They consist of stacked self-attention layers with residual connections. Each self-attention layer receives $n$ embeddings $\\{x_{i}\\}_{i=1}^{n}$ corresponding to unique input tokens, and outputs $n$ embeddings $\\{z_{i}\\}_{i=1}^{n}$, preserving the input dimensions. The $i$-th token is mapped via linear transformations to a key $k_{i}$, query $q_{i}$, and value $v_{i}$. The $i$-th output of the self-attention layer is given by weighting the values $v_{j}$ by the normalized dot product between the query $q_{i}$ and other keys $k_{j}$.

Equation: 
$$
z_{i}=\\sum_{j=1}^{n}\\mathsf{s o f}\\mathsf{t m a x}(\\{\\langle q_{i},k_{j^{\\prime}}\\rangle\\}_{j^{\\prime}=1}^{n})_{j}\\cdot v_{j}.
$$
"""


one_shot_response = "To derive the equation for the self-attention mechanism, we start by considering how Transformers process sequences. Each token embedding \( x_i \) is linearly transformed into a query (\( q_i \)), key (\( k_i \)), and value (\( v_i \)). The attention score for a token is computed as the dot product between its query and the keys of all other tokens, normalized using a softmax function. These normalized scores act as weights to aggregate the values (\( v_j \)) across all tokens, resulting in the output embedding (\( z_i \))."



for idx,md_file in enumerate(all_md_files):
    if idx < args.element_start or idx >= args.element_end:
        continue
    print(f"Processing paper {idx}, total time has been {time.time() - start_time} seconds")
    print(f'Num total generations: {num_total_generations}, Num added generations: {num_added_generations}')

    info_file = '/'.join(md_file.split('/')[:-1] + ['info.json'])
    with open(info_file, 'r') as f:
        info = json.load(f)
    title = info['title']

    if os.path.exists(md_file):
        # print(f'{idx}/{len(all_md_files)}')
        with open(md_file, "r") as f:
            content = f.read()
        og_content = content
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
        content = remove_related_work(content, headings)
        start_idx = start_idx_after_introduction
        # content = remove_all_but_three_sections(content, headings, start_idx)

        equation_indices = get_indices_of_equations(content)

        equation_indices = equation_indices[:5]
        print(equation_indices)
        visualize_data = False
        all_generated_texts = defaultdict(list)
        for j in range(5):
            print(f'Generating for {j}')
            for eqn_start, eqn_end in equation_indices:
                messages = [{"role": "user", "content": one_shot_prompt}, {"role": "assistant", "content": one_shot_response}]
                prefix = abstract + content[:eqn_start]
                equation = content[eqn_start:eqn_end]
                prompt = f'Context: {prefix}\nEquation: {equation}\nInstruction: Given the Context, please generate a chain of thought (less than 3 sentences) to explain how you arrived at the Equation. Do not put the given Equation in your response. Please use "we" instead of "I" and write as if you are adding text to the paper.'

                messages.append({"role": "user", "content": prompt})

                generated_text = generator2(messages)[0]['generated_text'][-1]['content']
                if j == 0:
                    print(generated_text)

                tokenized_prefix = perplexity_tokenizer(prefix, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_equation = perplexity_tokenizer(equation, return_tensors="pt").to(device)['input_ids'][0]
                tokenized_model_output = perplexity_tokenizer(generated_text, return_tensors="pt").to(device)['input_ids'][0]

                old_perplexity = get_perplexity(tokenized_prefix, tokenized_equation)
                new_perplexity = get_perplexity(tokenized_prefix, tokenized_equation, tokenized_model_output)
                if j == 0:
                    print(f"Old perplexity: {old_perplexity}, New perplexity: {new_perplexity}")
                num_total_generations += 1
                
                
                if new_perplexity < old_perplexity:
                    num_added_generations += 1
                    if visualize_data:
                        print(f'Adding to dataset: {generated_text}')
                        all_generated_texts[equation].append((generated_text, new_perplexity, old_perplexity))
                    else:
                        if j == 0:
                            print('Adding to dataset')

                        new_item = {
                            'prefix': prefix,
                            'equation': equation,
                            'model_output': generated_text,
                            'old_perplexity': old_perplexity,
                            'new_perplexity': new_perplexity
                        }

                        consolidated_output_path = f"{datadir}/arxiv_cot/incontext_sft_{mode}/elements_{args.element_start}_{args.element_end}.json"
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
                    if j == 0:
                        print('------------------')
        
        if visualize_data:
            best_completions = {}
            for equation, completions in all_generated_texts.items():
                if completions:  # Only process if there are completions for this equation
                    # Sort by new_perplexity (index 1 in tuple) and take the best one
                    index_val = og_content.find(equation)
                    before_equation = og_content[:index_val]
                    after_equation = og_content[index_val:]
                    all_completions = ""
                    for completion, new_perp, old_perp in completions:
                        all_completions += f'\n\n\n COT: {completion}, \n before perplexity: {old_perp}, after perplexity: {new_perp}\n\n\n'
                    og_content = before_equation + "<span style=\"color:blue\">\n\n Completions Start:" + all_completions + "\n Completions End \n\n</span>" + after_equation 
            # Save the markdown content to a file
            filename = md_file.split("/")[-2]
            output_md_path = f"/home/mprabhud/phd_projects/wild_o1/arxiv_cot/{filename}_coted.md"
            with open(output_md_path, 'w') as f:
                f.write(og_content)

        # # Save the best completions to file
        # consolidated_output_path = f"/home/mprabhud/phd_projects/wild_o1/arxiv_cot/incontext_sft_{args.mode}/elements_{args.element_start}_{args.element_end}.json"
        
        # output_data = []
        # for equation, (completion, new_perp, old_perp) in best_completions.items():
        #     output_data.append({
        #         'prefix': prefix,
        #         'equation': equation, 
        #         'model_output': completion,
        #         'old_perplexity': old_perp,
        #         'new_perplexity': new_perp
        #     })

        # # Write to file
        # with open(consolidated_output_path, 'w') as output_fp:
        #     json.dump(output_data, output_fp, indent=4)
        # print('done')
        
        
print('done')
