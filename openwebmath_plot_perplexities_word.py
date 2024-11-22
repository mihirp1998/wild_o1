from datasets import load_dataset, load_from_disk, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import matplotlib.pyplot as plt
import os

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def call_model(prompt):
    # Tokenize the prompt and move to device
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Get model predictions for each position
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # Remove last position's prediction as we don't have next token for it
        
    # Get the actual next tokens that occurred
    target_ids = input_ids[:, 1:]  # Shift right by 1 to get next tokens
    
    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = []
    
    # Get log prob of each actual token that appeared
    for i in range(target_ids.shape[1]):
        token_id = target_ids[0, i]
        token_log_prob = log_probs[0, i, token_id].item()
        token_log_probs.append(token_log_prob)
    
    # Get the input tokens for word boundary detection
    input_tokens = input_ids[0]
    final_answer = tokenizer.decode(input_tokens, skip_special_tokens=True)
    
    # Get individual tokens and their boundaries
    current_word = []
    current_word_log_prob = 0.0
    word_perplexity_pairs = []
    
    # Start from second token since we only have predictions for tokens after the first
    for i, token_id in enumerate(input_tokens[1:], 1):
        token = tokenizer.decode([token_id])
        current_word.append(token)
        current_word_log_prob += token_log_probs[i-1]  # i-1 because token_log_probs starts from first prediction
        
        # Check if token contains space or is the last token
        if ' ' in token or i == len(input_tokens) - 1:
            word = ''.join(current_word).strip()
            if word:  # Only add non-empty words
                # Convert log prob to perplexity
                word_perplexity = math.exp(-current_word_log_prob / len(current_word))
                word_perplexity_pairs.append((word, word_perplexity))
            current_word = []
            current_word_log_prob = 0.0
    
    return final_answer, word_perplexity_pairs[1:]

train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_0")

# Iterate through dataset (mocked here)
for i, item in enumerate(train_ds):
    if i >= 10:
        break
    metadata = json.loads(item["metadata"])
    math_score = metadata["extraction_info"]["math_score"]
    final_answer, word_perplexity_pairs = call_model(item["text"])

    # Create directory if it doesn't exist
    os.makedirs("perplexity_analysis_words", exist_ok=True)

    # Break word-perplexity pairs into chunks of 20
    chunk_size = 20
    chunks = [word_perplexity_pairs[i:i + chunk_size] for i in range(0, len(word_perplexity_pairs), chunk_size)]

    # Set up the subplots
    num_chunks = len(chunks)
    fig, axes = plt.subplots(num_chunks, 1, figsize=(12, 6 * num_chunks))  # Reduced height per subplot

    # Ensure axes is always iterable (handles single subplot case)
    if num_chunks == 1:
        axes = [axes]

    for idx, (chunk, ax) in enumerate(zip(chunks, axes)):
        words = [pair[0] for pair in chunk]
        perplexities = [pair[1] for pair in chunk]

        bars = ax.bar(range(len(words)), perplexities)

        # Customize each subplot
        ax.set_title(f"Word Perplexities for Words {idx * chunk_size + 1}-{(idx + 1) * chunk_size}")
        ax.set_xlabel("Words")
        ax.set_ylabel("Perplexity")
        ax.set_xticks(range(len(words)))
        # Escape special characters in labels
        escaped_words = [s.replace('$', '\$') for s in words]
        ax.set_xticklabels(escaped_words, rotation=60, ha="right")  # Increased rotation to 60 degrees

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

        # Adjust the position of the axes to make the plot area smaller and x-axis label area larger
        pos = ax.get_position()
        new_height = pos.height * 0.4  # Reduced plot height to 40% of original
        new_pos = [pos.x0, pos.y0 + (pos.height - new_height), pos.width, new_height]
        ax.set_position(new_pos)

    # Adjust layout with more bottom space
    plt.subplots_adjust(bottom=0.5, hspace=0.7)  # Increased bottom margin to 50% and adjusted hspace

    # Save the plot
    plt.savefig(f"perplexity_analysis_words/stacked_perplexities_{i}.png")
    plt.close()