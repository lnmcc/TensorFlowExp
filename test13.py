import tensorflow as tf
import json
import string
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from collections import OrderedDict
from bs4 import BeautifulSoup
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = open("irish-lyrics-eof.txt").read()
corpus = data.lower().split("\n")
# print(corpus)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
# print(tokenizer.word_index)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)
print(input_sequences[:5])
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
print(labels)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 8))
model.add(
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(max_sequence_len - 1, return_sequences="True")
    )
)
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sequence_len - 1)))
model.add(tf.keras.layers.Dense(total_words, activation="Softmax"))

epochs = 1000
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(xs, ys, epochs=epochs, verbose=1)

seed_tex = "sweet jeremy saw dublin"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_tex])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    # predicated = model.predict_classes(token_list, verbose=0)
    predicated = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicated:
            output_word = word
            break
    seed_tex += " " + output_word

print(seed_tex)

print(history.history.keys())

acc = history.history["accuracy"]
loss = history.history["loss"]

epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.show()
