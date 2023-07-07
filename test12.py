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


data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \n His father died and made him a man again \n Left him a farm and ten acres of ground. \n He gave a grand party for friends and relations \n Who didnt forget him when come to the wall, \n And if youll but listen Ill make your eyes glisten \n Of the rows and the ructions of Lanigan’s Ball. \n Myself to be sure got free invitation, \n For all the nice girls and boys I might ask, \n And just in a minute both friends and relations \n Were dancing round merry as bees round a cask. \n Judy ODaly, that nice little milliner, \n She tipped me a wink for to give her a call, \n And I soon arrived with Peggy McGilligan \n Just in time for Lanigans Ball."
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
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sequence_len - 1)))
model.add(tf.keras.layers.Dense(total_words, activation="Softmax"))

epochs = 1500
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(xs, ys, epochs=epochs, verbose=1)

seed_text = "in the town of athy"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences(
    [token_list], maxlen=max_sequence_len - 1, padding="pre"
)
predicated = np.argmax(model.predict(token_list), axis=-1)
print(predicated)

for word,index in tokenizer.word_index.items():
    if index == predicated:
        print(word)
        break

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
