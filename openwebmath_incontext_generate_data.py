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
parser.add_argument("--chunk_start", type=int, default=0)
parser.add_argument("--chunk_end", type=int, default=1)
parser.add_argument("--element_start", type=int, default=0)
parser.add_argument("--element_end", type=int, default=1000)
parser.add_argument("--mode", type=str, default='train')

# Parse arguments
args = parser.parse_args()

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


generate_prompt = """
{text}
Instruction: Given the above text, please identify the sentence that is most difficult to understand and then provide a chain of thought to explain that sentence (less than 200 tokens).
Please ignore anything that is not relevant, such as metadata. The sentence should be an exact copy of the original text.
Please answer in the format: Sentence to explain: "<sentence>"\nChain of thought: Step 1: <step 1>\nStep 2: <step 2>\nStep 3: <step 3>\n
Sentence to explain: 
"""

mode = args.mode

for chunk_num in range(int(args.chunk_start), int(args.chunk_end)):

    train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_{chunk_num}")

    start_time = time.time()

    if not os.path.exists(f"/grogu/user/lilic/filtered_openwebmath/incontext_sft_{mode}"):
        os.makedirs(f"/grogu/user/lilic/filtered_openwebmath/incontext_sft_{mode}")

    for i, item in enumerate(train_ds):
        if i < args.element_start or i >= args.element_end:
            continue
        if i % 10 == 0:
            print(f"Processing element {i}, total time has been {time.time() - start_time} seconds")

        metadata = json.loads(item["metadata"])
        math_score = metadata["extraction_info"]["math_score"]

        prompt = generate_prompt.format(text=item["text"])

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(inputs["input_ids"].shape)
        if inputs["input_ids"].shape[1] > 512:
            continue

        for _ in range(5):
            output = model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + 200,  # Adjust as needed
                temperature=1.5,
                num_return_sequences=2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                top_p=0.9,
                top_k=50
            )

            decoded_output = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            index_of_chain_of_thought = decoded_output.find("Chain of thought:")
            if index_of_chain_of_thought != -1:
                model_output_sentence = decoded_output[:index_of_chain_of_thought].strip()
                model_output_cot = decoded_output[index_of_chain_of_thought:].strip()
            else:
                model_output_sentence = ""
                model_output_cot = decoded_output

            new_item = {
                'text': item['text'],
                'model_output_sentence': model_output_sentence,
                'model_output_cot': model_output_cot
            }
            print(model_output_sentence)
            print('------------------')
            print(model_output_cot)
            print('#################')


            consolidated_output_path = f"/grogu/user/lilic/filtered_openwebmath/incontext_sft_{mode}/chunk_{chunk_num}_elements_{args.element_start}_{args.element_end}.json"
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