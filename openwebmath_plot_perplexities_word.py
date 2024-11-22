from datasets import load_dataset, load_from_disk, Dataset
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import matplotlib.pyplot as plt
import os
import html

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
        # Escape special characters while preserving unicode
        escaped_word = html.escape(word)
        color = f"rgba(255, 0, 0, {opacity})"  # Highlight with varying red opacity
        html_content += f'<span style="background-color: {color}; padding: 2px; margin: 2px;">{escaped_word}</span> '
    html_content += "</div>"
    
    # Display in Jupyter Notebook
    display(HTML(html_content))
    
    # Save to HTML file if output_file is provided
    if output_file:
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            padding: 20px;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>'''
        
        if append and os.path.exists(output_file):
            with open(output_file, "r", encoding='utf-8') as file:
                existing_content = file.read()
                # Insert new content before the closing tags
                if "</body></html>" in existing_content:
                    existing_content = existing_content.replace("</body></html>", f"{html_content}</body></html>")
                else:
                    existing_content += html_content
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(existing_content)
        else:
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(html_template.format(content=html_content))

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
        token = tokenizer.decode([token_id], skip_special_tokens=True)
        current_word.append(token)
        current_word_log_prob += token_log_probs[i-1]  # i-1 because token_log_probs starts from first prediction
        
        # Check if next token starts with space (meaning current token completes a word) or if this is the last token
        next_token = tokenizer.decode([input_tokens[i+1]], skip_special_tokens=True) if i < len(input_tokens) - 1 else None
        if (next_token and next_token.startswith(' ')) or i == len(input_tokens) - 1:
            word = ''.join(current_word).strip()
            if word:  # Only add non-empty words
                # Convert log prob to perplexity
                # word_perplexity = math.exp(-current_word_log_prob / len(current_word))
                word_perplexity = -current_word_log_prob / len(current_word)
                word_perplexity_pairs.append((word, word_perplexity))
            current_word = []
            current_word_log_prob = 0.0
    
    return final_answer, word_perplexity_pairs[1:]

train_ds = load_from_disk(f"/grogu/user/lilic/filtered_openwebmath/train/chunk_0")

# Iterate through dataset (mocked here)
os.makedirs("perplexity_analysis_words", exist_ok=True)
output_file = "perplexity_analysis_words/highlighted_text.html"

# Start with a fresh file
if os.path.exists(output_file):
    os.remove(output_file)

for i, item in enumerate(train_ds):
    if i >= 10:  # Process first 10 examples
        break
    metadata = json.loads(item["metadata"])
    math_score = metadata["extraction_info"]["math_score"]
    final_answer, word_perplexity_pairs = call_model(item["text"])

    # Example usage
    words = [pair[0] for pair in word_perplexity_pairs]
    values = [pair[1] for pair in word_perplexity_pairs]

    # Append each example to the same file
    highlight_text(words, values, output_file=output_file, append=i > 0)
    print(f"Processed example {i+1}/10")