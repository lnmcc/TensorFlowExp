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

# import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


stopwords = ["a", ..., "yourselves"]
table = str.maketrans("", "", string.punctuation)

with open("sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentence = item["headline"].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])

training_size = 23000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 20000
max_length = 10
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# print(word_index)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


wc = tokenizer.word_counts
# print(wc)

newlist = OrderedDict(sorted(wc.items(), key=lambda t: t[1], reverse=True))
# print(newlist)
xs = []
ys = []
curr_x = 1
for item in newlist:
    xs.append(curr_x)
    curr_x = curr_x + 1
    ys.append(newlist[item])

plt.plot(xs, ys)
# plt.axis([300,10000,0,100])
# plt.show()


training_sentences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sentences, padding=padding_type)

testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, padding=padding_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# print(training_padded)
# print(training_labels)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(20000, 7),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(
            8, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
        ),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False
)

model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
model.summary()
model.fit(training_padded, training_labels, epochs=10)
# model.evaluate(testing_padded, testing_labels)

new_index = word_index["new"]

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
print(reverse_word_index[new_index])
print(weights[new_index])

out_v = io.open("vecs.tsv", "w", encoding="utf-8")
out_m = io.open("meta.tsv", "w", encoding="utf-8")
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write("\t".join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
