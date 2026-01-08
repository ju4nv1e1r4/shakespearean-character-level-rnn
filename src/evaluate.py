import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
import math

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "shakespeare_best.keras")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.json")
SEQ_LENGTH = 100
BATCH_SIZE = 64

print("--- 1. Loading artifacts ---")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Maybe you need to train the model first?")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(TOKENIZER_PATH) as f:
    tokenizer_json = f.read()
tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
print("Tokenizer loaded.")

print("\n--- 2. Preparing Test Set (Hold-out Set) ---")
shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    text = f.read()

[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
dataset_size = len(encoded)

train_size = dataset_size * 90 // 100
test_data = encoded[train_size:]

print(f"Total test size: {dataset_size} chars.")
print(f"Set test size: {len(test_data)} chars (10%).")

def create_dataset(sequence):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(SEQ_LENGTH + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(SEQ_LENGTH + 1))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    return ds.prefetch(1)

test_dataset = create_dataset(test_data)

print("\n--- 3. Evaluating Model---")
results = model.evaluate(test_dataset, return_dict=True, verbose=1)

loss = results['loss']

accuracy = next((v for k, v in results.items() if 'acc' in k), 0)

# e^loss -> Perplexity calc
perplexity = math.exp(loss)

print("\n" + "="*30)
print("EVALUATION REPORT")
print("="*30)
print(f"Cross Entropy Loss:  {loss:.4f}")
print(f"Accuracy (Top-1):    {accuracy*100:.2f}%")
print(f"Perplexity:        {perplexity:.2f}")
print("-" * 30)

if perplexity < 5:
    print("STATUS: Excellent!")
elif perplexity < 10:
    print("STATUS: Good enought.")
else:
    print("STATUS: The model is confused.")

print("="*30)

print("\nCalculating Top-5 Accuracy...")

for X_batch, y_batch in test_dataset.take(1):
    y_pred_proba = model.predict(X_batch, verbose=0) # shape: (batch, seq_len, vocab_size)
    
    # Keras metrics func
    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    m.update_state(y_batch, y_pred_proba)
    top5 = m.result().numpy()
    
    print(f"Top-5 Accuracy on a random batch: {top5*100:.2f}%")
    # This means: "In X% of cases, the correct character was among the top 5 options in the template."
