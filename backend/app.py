from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import traceback
import os

app = Flask(__name__)
CORS(app)

# --- LAZY LOADING SETUP ---
# Initialize model and tokenizer as None. They will be loaded on the first request.
model = None
tokenizer = None
model_name = "Amitabhdas/code-summarizer-python"

def load_model():
    """Checks if the model is loaded, and if not, downloads and initializes it."""
    global model, tokenizer
    if model is None or tokenizer is None:
        # The TRANSFORMERS_CACHE environment variable will direct this to /tmp
        print("Model not loaded. Initializing model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully.")

# --- Your Existing Functions (Unchanged) ---
def summarize_code_with_attention(code_snippet):
    inputs = tokenizer(
        code_snippet, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, max_length=150, num_beams=4, early_stopping=True, output_attentions=True, return_dict_in_generate=True
        )
    summary = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return summary, outputs.encoder_attentions, inputs

def tokens_to_words(tokens):
    words, current_word = [], ""
    for token in tokens:
        if token.startswith(" "):
            if current_word: words.append(current_word)
            current_word = token[1:]
        else:
            current_word += token
    if current_word: words.append(current_word)
    return words

def compute_word_importance(attention, inputs):
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    word_tokens = tokens_to_words(tokens)
    attention_matrix = torch.cat(attention).mean(dim=1)[0, :, :]
    word_importance = [0] * len(word_tokens)
    token_idx = 0
    for i, word in enumerate(word_tokens):
        num_tokens = len(tokenizer.tokenize(word, add_special_tokens=False))
        if num_tokens > 0:
            word_importance[i] = attention_matrix[token_idx:token_idx + num_tokens, :].mean()
        token_idx += num_tokens
    max_importance = max(word_importance) if word_importance else 1
    normalized_importance = [imp / max_importance for imp in word_importance]
    return sorted(zip(word_tokens, normalized_importance), key=lambda x: x[1], reverse=True)[:10]

def generate_counterfactuals(code):
    variations = {}
    variations['remove_else'] = re.sub(r'else:', 'if True:', code, 1)
    variations['change_comparison'] = re.sub(r'==', '!=', code, 1)
    return variations

# --- API Routes ---
@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        load_model()  # Ensure model is loaded before processing
        data = request.json
        code = data.get('code', '')
        if not code: return jsonify({'error': 'No code provided'}), 400
        summary, attention, inputs = summarize_code_with_attention(code)
        top_words = compute_word_importance(attention, inputs)
        result = {'summary': summary, 'topWords': [{'word': w, 'score': float(s)} for w, s in top_words]}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/counterfactuals', methods=['POST'])
def counterfactuals():
    try:
        load_model()  # Ensure model is loaded before processing
        data = request.json
        code = data.get('code', '')
        if not code: return jsonify({'error': 'No code provided'}), 400
        cf_variations = generate_counterfactuals(code)
        results = []
        for label, cf_code in cf_variations.items():
            summary, attention, inputs = summarize_code_with_attention(cf_code)
            top_words = compute_word_importance(attention, inputs)
            results.append({'label': label, 'code': cf_code, 'summary': summary, 'topWords': [{'word': w, 'score': float(s)} for w, s in top_words]})
        return jsonify({'counterfactuals': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})