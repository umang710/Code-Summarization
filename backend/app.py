from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer once at startup
model_name = "Amitabhdas/code-summarizer-python"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Move model to GPU if available

# Import your existing functions (paste.txt)
# Summarization with attention extraction
def summarize_code_with_attention(code_snippet):
    inputs = tokenizer(
        code_snippet,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)  # Move inputs to the same device as model
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True,
            output_attentions=True,
            return_dict_in_generate=True
        )
    summary = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    attention = outputs.encoder_attentions
    return summary, attention, inputs

# Subword token to word reconstruction
def tokens_to_words(tokens):
    words = []
    current_word = ""
    mapping = []
    for i, token in enumerate(tokens):
        if token.startswith("▁"):
            if current_word:
                words.append(current_word)
            current_word = token.lstrip("▁")
            mapping.append(len(words))
        else:
            current_word += token
            mapping.append(len(words) - 1 if words else 0)
    if current_word:
        words.append(current_word)
    return words, mapping

# Compute top word importance scores
def compute_word_importance(attention_weights, inputs):
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    non_pad_mask = inputs.attention_mask[0].bool()
    all_layers_attention = [layer.mean(dim=1) for layer in attention_weights]
    all_attention = torch.stack(all_layers_attention).mean(dim=0).squeeze()
    filtered_attention = all_attention[non_pad_mask][:, non_pad_mask]
    filtered_tokens = [token for i, token in enumerate(tokens) if non_pad_mask[i]]
    words, token_to_word_map = tokens_to_words(filtered_tokens)
    token_importance = filtered_attention.sum(dim=0).cpu().numpy()  # Move to CPU for numpy conversion
    word_scores, word_counts = {}, {}
    for idx, word_idx in enumerate(token_to_word_map):
        word = words[word_idx]
        word_scores[word] = word_scores.get(word, 0) + token_importance[idx]
        word_counts[word] = word_counts.get(word, 0) + 1
    word_importance = [(word, word_scores[word] / word_counts[word]) for word in words]
    word_importance = list(dict.fromkeys(word_importance))  # dedup
    word_importance.sort(key=lambda x: x[1], reverse=True)
    return word_importance[:10]

# Dynamically generate counterfactual code variations
def generate_counterfactuals(original_code):
    counterfactuals = {}

    cf1 = re.sub(r'def factorial', 'def compute_factorial', original_code)
    cf1 = re.sub(r'\bfactorial\b', 'compute_factorial', cf1)
    counterfactuals["rename_function"] = cf1

    cf2 = re.sub(r'if n == 0', 'if n <= 1', original_code)
    counterfactuals["change_base_case"] = cf2

    lines = original_code.split('\n')
    for i, line in enumerate(lines):
        if "def " in line:
            indent = ' ' * (len(line) - len(line.lstrip()) + 4)
            lines.insert(i + 1, indent + 'print("Calculating factorial")')
            break
    cf3 = '\n'.join(lines)
    counterfactuals["add_print"] = cf3

    cf4 = """def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result"""
    counterfactuals["iterative"] = cf4

    cf5 = """def factorial(n):
    if n != 0:
        return n * factorial(n-1)
    else:
        return 1"""
    counterfactuals["reverse_condition"] = cf5

    return counterfactuals

# API endpoints
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Generate summary and attention
        summary, attention, inputs = summarize_code_with_attention(code)
        
        # Get top important words
        top_words = compute_word_importance(attention, inputs)
        
        # Format response
        result = {
            'summary': summary,
            'topWords': [{'word': word, 'score': float(score)} for word, score in top_words]
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in summarize: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/counterfactuals', methods=['POST'])
def counterfactuals():
    try:
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Generate counterfactuals
        cf_variations = generate_counterfactuals(code)
        
        results = []
        
        for label, cf_code in cf_variations.items():
            # Generate summary and attention for each counterfactual
            summary, attention, inputs = summarize_code_with_attention(cf_code)
            top_words = compute_word_importance(attention, inputs)
            
            results.append({
                'label': label,
                'code': cf_code,
                'summary': summary,
                'topWords': [{'word': word, 'score': float(score)} for word, score in top_words]
            })
        
        return jsonify({'counterfactuals': results})
    
    except Exception as e:
        print(f"Error in counterfactuals: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)