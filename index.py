from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("-----------------------------------------------------------------------------")

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data,
                             test_label) = imdb.load_data(num_words=10000)

##print(f"Test {len(train_data)} {len(train_labels)} {len(test_data)}")

word_index = imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "? ") for i in text])


# print(decode_review(test_data[3]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding="post",
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding="post",
                                                       maxlen=256)
# Modelo
vocabSize = 1000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocabSize, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.selu))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
