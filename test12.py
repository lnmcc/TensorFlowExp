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
print(corpus)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
