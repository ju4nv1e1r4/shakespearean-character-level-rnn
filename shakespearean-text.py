import tensorflow as tf
from tensorflow import keras
import numpy as np

shakespeare_url = "https://homl.info/shakespeare"
shakespeare_path = keras.utils.get_file("shakespeare.txt", shakespeare_url)

with open(shakespeare_path) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

print("Vocabulary size:", tokenizer.texts_to_sequences(["First"]))

print("Inverse vocabulary:", tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))

max_id = len(tokenizer.word_index)
print("Max ID:", max_id)
dataset_size = tokenizer.document_count
print("Dataset size:", dataset_size)

print("---")

encoded = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
encoded = encoded.flatten()
print("Encoded shape:", encoded.shape)

dataset_size = len(encoded)
print("Dataset size:", dataset_size)

train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, y_batch: (tf.one_hot(X_batch, depth=max_id), y_batch)
)

dataset = dataset.prefetch(1)

model = keras.models.Sequential([
    keras.Input(shape=(None, max_id)),
    keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax")),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model.keras',
    save_best_only=True
)

history = model.fit(
    dataset,
    epochs=20,
    callbacks=[checkpoint],
)

def preprocess(text):
    X = tokenizer.texts_to_sequences(text)
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are yo"])  # shape: (1, sequence_length, vocab_size)
Y_proba = model.predict(X_new)
Y_pred = tf.argmax(Y_proba, axis=-1).numpy()

predicted_char_id = Y_pred[0, -1] + 1
predicted_char = tokenizer.sequences_to_texts([[predicted_char_id]])[0]
print(predicted_char)