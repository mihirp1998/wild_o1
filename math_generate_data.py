import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv
import argparse

HF_TOKEN = "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"
HF_TOKEN[:3]+'...'

parser = argparse.ArgumentParser(description="Run model on MATH dataset.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model checkpoint or Hugging Face model name.")
parser.add_argument("--consolidated_output_path", type=str, required=True, help="Path to the output JSON file.")
parser.add_argument("--subdir", type=str, required=True, help="Directory path for the MATH dataset subset (e.g., prealgebra).")

args = parser.parse_args()

# Load the Llama-3.2-1B-Instruct model and tokenizer
# model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the exact model name if needed
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = args.model_name # "/grogu/user/lilic/hendrycks-sft/0042/checkpoint-1000"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

rootdir = "/grogu/user/lilic/MATH/MATH/train"

def call_model(problem, answer):
    '''
    Given a problem and its answer, returns the model's chain of thought explanation.
    '''
    # New prompt format
    test_question = (
        f"\nProblem: {problem}\nAnswer: {answer}\n"
        "Given the above math problem and answer, can you produce a chain of thought (less than 200 tokens) to explain how you arrived at the answer?"
    )
    prompt = test_question

    # Tokenize the prompt and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + 200,  # Adjust as needed
            temperature=1.0,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get the generated tokens (excluding the prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    # Decode the generated tokens
    final_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return final_answer

def run(max=-1):
    outputs = []
    answers = []
    types = []
    levels = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    for subdir, dirs, files in os.walk(rootdir):
        # if subdir == '/grogu/user/lilic/MATH/MATH/train/prealgebra':
        if subdir == args.subdir:
            for file in files:
                fnames_list.append(os.path.join(subdir, file))
                with open(os.path.join(subdir, file), 'r') as fp:
                    try:
                        problem_data = json.load(fp)
                    except Exception as e:
                        print(f"Error loading JSON from {file}", e)
                        raise e
                    prob_level = problem_data["level"]
                    prob_type = problem_data["type"]
                    try:
                        prob_level = int(prob_level.split("Level ")[1])
                    except:
                        prob_level = None

                    print(problem_data["problem"])
                    print('-----------------------------------')
                    if total > 10:
                        break

                    # model_output = call_model(train_prompt, problem_data["problem"])
                    # model_output = call_model(problem_data["problem"], remove_boxed(last_boxed_only_string(problem_data["solution"])))


                    # # Path to the consolidated output file
                    # consolidated_output_path = args.consolidated_output_path # "/grogu/user/lilic/MATH/MATH/sft_train/prealgebra_data4.json"

                    # # Check if the file exists to either initialize it as an empty list or load existing data
                    # if os.path.exists(consolidated_output_path):
                    #     with open(consolidated_output_path, 'r') as output_fp:
                    #         try:
                    #             all_data = json.load(output_fp)
                    #         except json.JSONDecodeError:
                    #             all_data = []  # Start with an empty list if the file is corrupted or empty
                    # else:
                    #     all_data = []  # Start with an empty list if the file does not exist

                    # # Prepare the new entry
                    # output_data = {
                    #     "problem": problem_data["problem"],
                    #     "solution": problem_data["solution"],
                    #     "model_output": model_output,
                    #     "level": problem_data["level"],
                    #     "type": problem_data["type"]
                    # }

                    # # Append the new entry to the list
                    # all_data.append(output_data)

                    # # Write the updated list back to the single file
                    # with open(consolidated_output_path, 'w') as output_fp:
                    #     json.dump(all_data, output_fp, indent=4)

                    # print("Problem:")
                    # print(problem_data["problem"])
                    # print("Model output:")
                    # print(model_output)
                    # print("Correct answer:")
                    # print(problem_data["solution"])
                    # print("--------------------------------------------")

                    total += 1
                    print(total)
                    print('---')

                if max > 0 and total >= max:
                    break
            if max > 0 and total >= max:
                break

if __name__ == "__main__":
    # You can set max to a positive integer to limit the number of problems processed (useful for testing)
    run(max=-1)  # Set max to -1 to process all problems

