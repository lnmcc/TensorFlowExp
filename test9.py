import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tfds.list_builders()

#imdb_sentences = []
#train_data = tfds.as_numpy(tfds.load("imdb_reviews", split="train"))
#for item in train_data:
#    print(item)
#    imdb_sentences.append(str(item["text"]))
