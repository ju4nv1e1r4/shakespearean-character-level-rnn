from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

MODEL_PATH = "artifacts/shakespeare_best.keras"
TOKENIZER_PATH = "artifacts/tokenizer.json"

app = FastAPI(title="Shakespeare Language Model API")

model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str = "The king"
    temperature: float = 0.6
    length: int = 200

def load_artifacts():
    global model, tokenizer
    print("Carregando modelo e tokenizador...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH) as f:
            tokenizer_json = f.read()
        tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        print("Artefatos carregados com sucesso!")
    except Exception as e:
        print(f"Erro fatal ao carregar artefatos: {e}")

load_artifacts()

def preprocess(text):
    X = np.array(tokenizer.texts_to_sequences([text])) - 1
    return tf.convert_to_tensor(X, dtype=tf.int32)

def next_char(text, temperature=1.0):
    X_new = preprocess(text)
    y_proba = model.predict(X_new, verbose=0)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    generated_text = request.prompt
    current_input = request.prompt

    for _ in range(request.length):
        next_c = next_char(current_input, request.temperature)
        generated_text += next_c
        current_input += next_c
        
    return {
        "prompt": request.prompt,
        "generated_text": generated_text,
        "model_version": "v1"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0}