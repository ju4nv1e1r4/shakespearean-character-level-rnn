import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "shakespeare_best.keras")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.json")


print("Loading model and tokenizer...")
model = keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH) as f:
    tokenizer_json = f.read()
tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def preprocess(text):
    X = np.array(tokenizer.texts_to_sequences([text])) - 1
    return tf.convert_to_tensor(X, dtype=tf.int32)

def next_char(text, temperature=1.0):
    """
    temperature: 
      - Close to 0: Conservative, repetitive, grammatically correct.
      - Close to 1: Creative, varied.
      - <1: Chaotic, incoherent and creating words (HALLUCINATION ALERT).
    """
    X_new = preprocess(text)
    y_proba = model.predict(X_new, verbose=0)[0, -1:, :]
    
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(prompt, n_chars=200, temperature=0.7):
    print(f"--- Generating {n_chars} chars with temperature {temperature} ---")
    input_prompt = f"{prompt}...\n"
    print(input_prompt, end="", flush=True)
    
    current_text = prompt
    for _ in range(n_chars):
        next_c = next_char(current_text, temperature)
        print(next_c, end="", flush=True)
        current_text += next_c
    print("\n\n--- The End ---")

# let's test
initial_prompt = input("Enter a shakespearean prompt: ")
complete_text(initial_prompt, n_chars=300, temperature=0.6)
