import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 10
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("--- Loading data ---")
shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])
max_id = len(tokenizer.word_index)

tokenizer_json = tokenizer.to_json()
with open(os.path.join(ARTIFACTS_DIR, 'tokenizer.json'), 'w') as f:
    f.write(tokenizer_json)
print("Tokenizer saved in artifacts/tokenizer.json")

[encoded] = np.array(tokenizer.texts_to_sequences([text])) - 1
dataset_size = len(encoded)
train_size = dataset_size * 90 // 100

def create_dataset(sequence):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(SEQ_LENGTH + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(SEQ_LENGTH + 1))
    ds = ds.shuffle(10000).batch(BATCH_SIZE)
    ds = ds.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
    return ds.prefetch(1)

train_set = create_dataset(encoded[:train_size])
# Would be nice to have a validation set
# valid_set = create_dataset(encoded[train_size:]) 

strategy = tf.distribute.MirroredStrategy() # gpu strategy
with strategy.scope():
    model = keras.models.Sequential([
        keras.Input(shape=[None], dtype=tf.int32),
        keras.layers.Embedding(max_id + 1, 64),
        keras.layers.GRU(128, return_sequences=True, dropout=0.2), # GRU is lighter than LSTM
        keras.layers.GRU(128, return_sequences=True, dropout=0.2),
        keras.layers.TimeDistributed(keras.layers.Dense(max_id + 1, activation="softmax"))
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(ARTIFACTS_DIR, "shakespeare_best.keras"),
    save_best_only=True, # with validation set we could monitor val_loss
    monitor='loss' 
)

print("--- Train is starting ---")
model.fit(train_set, epochs=EPOCHS, callbacks=[checkpoint_cb])
print("Model saved in artifacts/shakespeare_best.keras!")
