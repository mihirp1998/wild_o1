from datasets import load_dataset, load_from_disk, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

def highlight_text(words, values, output_file=None, append=False):
    """
    Highlights words with different opacities based on the values provided.
    
    Args:
        words (list): List of words to be highlighted.
        values (list): List of numerical values corresponding to the words. 
                       The values determine the opacity of the highlight (0 to 1).
        output_file (str): If provided, saves the output HTML to this file.
        append (bool): If True, appends to existing file instead of overwriting.
    """
    # Normalize the values to be between 0 and 1
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Create the HTML string for this example
    html_content = "<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ccc;'>"
    for word, opacity in zip(words, normalized_values):
        color = f"rgba(255, 0, 0, {opacity})"  # Highlight with varying red opacity
        html_content += f'<span style="background-color: {color}; padding: 2px; margin: 2px;">{word}</span> '
    html_content += "</div>"
    
    # Display in Jupyter Notebook
    display(HTML(html_content))
    
    # Save to HTML file if output_file is provided
    if output_file:
        if append and os.path.exists(output_file):
            with open(output_file, "r") as file:
                existing_content = file.read()
                # Insert new content before the closing tags
                if "</body></html>" in existing_content:
                    existing_content = existing_content.replace("</body></html>", f"{html_content}</body></html>")
                else:
                    existing_content += html_content
            with open(output_file, "w") as file:
                file.write(existing_content)
        else:
            with open(output_file, "w") as file:
                file.write(f"<html><head><style>body {{ font-family: Arial, sans-serif; padding: 20px; }}</style></head><body>{html_content}</body></html>")


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
    
    # Get the input tokens for sentence boundary detection
    input_tokens = input_ids[0]
    final_answer = tokenizer.decode(input_tokens, skip_special_tokens=True)
    
    # Get individual tokens and their boundaries
    tokens = []
    current_sentence = []
    current_sentence_log_prob = 0.0
    sentence_perplexity_pairs = []
    
    # Start from second token since we only have predictions for tokens after the first
    for i, token_id in enumerate(input_tokens[1:], 1):
        token = tokenizer.decode([token_id])
        current_sentence.append(token)
        current_sentence_log_prob += token_log_probs[i-1]  # i-1 because token_log_probs starts from first prediction
        
        # Check if token ends with sentence boundary markers
        if token.endswith('\n') or any(token.rstrip().endswith(p) for p in ['.', '!', '?', '\n']):
            sentence = ''.join(current_sentence).strip()
            if sentence:  # Only add non-empty sentences
                # Convert log prob to perplexity
                # sentence_perplexity = math.exp(-current_sentence_log_prob / len(current_sentence))
                sentence_perplexity = -current_sentence_log_prob / len(current_sentence)
                sentence_perplexity_pairs.append((sentence, sentence_perplexity))
            current_sentence = []
            current_sentence_log_prob = 0.0
    
    # Add the last sentence if exists
    if current_sentence:
        sentence = ''.join(current_sentence).strip()
        if sentence:
            sentence_perplexity = math.exp(-current_sentence_log_prob / len(current_sentence))
            sentence_perplexity_pairs.append((sentence, sentence_perplexity))
    
    return final_answer, sentence_perplexity_pairs[1:]

train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_0")

# Iterate through dataset (mocked here)
os.makedirs("perplexity_analysis", exist_ok=True)
output_file = "perplexity_analysis/highlighted_text.html"

# Start with a fresh file
if os.path.exists(output_file):
    os.remove(output_file)

# Iterate through dataset (mocked here)
for i, item in enumerate(train_ds):
    if i >= 10:
        break
    metadata = json.loads(item["metadata"])
    math_score = metadata["extraction_info"]["math_score"]
    final_answer, sentence_perplexity_pairs = call_model(item["text"])

    # Create directory if it doesn't exist
    os.makedirs("perplexity_analysis", exist_ok=True)
     # Example usage
    sentences = [pair[0] for pair in sentence_perplexity_pairs]
    values = [pair[1] for pair in sentence_perplexity_pairs]

    # Append each example to the same file
    highlight_text(sentences, values, output_file=output_file, append=i > 0)
    print(f"Processed example {i+1}/10")