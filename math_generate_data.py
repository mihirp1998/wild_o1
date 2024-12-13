import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv
import argparse
import transformers
import glob
import time

HF_TOKEN = "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"
HF_TOKEN[:3]+'...'

parser = argparse.ArgumentParser(description="Run model on MATH dataset.")
parser.add_argument("--model_name", type=str, required=True, help="Path to the model checkpoint or Hugging Face model name.")
parser.add_argument("--consolidated_output_path", type=str, required=True, help="Path to the output JSON file.")
parser.add_argument("--subdir", type=str, required=True, help="Directory path for the MATH dataset subset (e.g., prealgebra).")
parser.add_argument("--num_generations_per_problem", type=int, default=1, help="Number of generations per problem.")

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

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    device=device,
    do_sample=True,
    num_return_sequences=args.num_generations_per_problem
)

def call_model(problem, answer):
    '''
    Given a problem and its answer, returns the model's chain of thought explanation.
    '''
    prompt = (
        f"\nProblem: {problem}\nAnswer: {answer}\n"
        "Instruction: Please produce a chain of thought (less than 200 tokens) to explain how you arrived at the answer. Please put the final answer in \\boxed{}."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    messages = [
        {"role": "user", "content": prompt}
    ]

    outputs = []
    with torch.no_grad():
        generations = generator(messages)
        for gen in generations:
            outputs.append(gen['generated_text'][-1]['content'])
    
    return outputs

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

    start_time = time.time()

    for subdir, dirs, files in os.walk(rootdir):
        # if subdir == '/grogu/user/lilic/MATH/MATH/train/prealgebra':
        if subdir == args.subdir:
            for file in files:
                fnames_list.append(os.path.join(subdir, file))
                num_overall = len(glob.glob(subdir + "/*"))
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

                    all_model_outputs = call_model(problem_data["problem"], remove_boxed(last_boxed_only_string(problem_data["solution"])))
                    for j in range(args.num_generations_per_problem):
                        model_output = all_model_outputs[j]

                        generated_answer = remove_boxed(last_boxed_only_string(model_output))
                        ground_truth_answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
                        if generated_answer == ground_truth_answer:
                            # Path to the consolidated output file
                            category = subdir.split("/")[-1]
                            output_filename = args.consolidated_output_path + f"/{category}.json"

                            # Check if the file exists to either initialize it as an empty list or load existing data
                            if os.path.exists(output_filename):
                                with open(output_filename, 'r') as output_fp:
                                    try:
                                        all_data = json.load(output_fp)
                                    except json.JSONDecodeError:
                                        all_data = []  # Start with an empty list if the file is corrupted or empty
                            else:
                                all_data = []  # Start with an empty list if the file does not exist
                                os.makedirs(os.path.dirname(args.consolidated_output_path), exist_ok=True)

                            # Prepare the new entry
                            output_data = {
                                "problem": problem_data["problem"],
                                "solution": problem_data["solution"],
                                "model_output": model_output,
                                "level": problem_data["level"],
                                "type": problem_data["type"]
                            }

                            # Append the new entry to the list
                            all_data.append(output_data)

                            # Write the updated list back to the single file
                            with open(output_filename, 'w') as output_fp:
                                json.dump(all_data, output_fp, indent=4)

                        if j == 0:
                            print("Problem:")
                            print(problem_data["problem"])
                            print("Model output:")
                            print(model_output)
                            print("Correct answer:")
                            print(problem_data["solution"])
                            print("-------------------------------------------------------------------")

                        total += 1
                        print(f"{total}/{num_overall * args.num_generations_per_problem} processed. time has been {time.time() - start_time} seconds")
                        print('---')

                if max > 0 and total >= max:
                    break
            if max > 0 and total >= max:
                break

if __name__ == "__main__":
    # You can set max to a positive integer to limit the number of problems processed (useful for testing)
    run(max=-1)  # Set max to -1 to process all problems

