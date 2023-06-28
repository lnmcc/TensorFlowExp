import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#print(tfds.list_builders())

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load("imdb_reviews", split="train"))
for item in train_data:
    imdb_sentences.append(str(item["text"]))

(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True,
)

encoder=info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))
#print(encoder.subwords)

sample_string='Today is a sunny day'
encoder_string=encoder.encode(sample_string)
print('Encoded string is {}'.format(encoder_string))
print(encoder.subwords[6426])
print(encoder.subwords[4868])

orignal_string=encoder.decode(encoder_string)
print(orignal_string)