#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

n_top_words = 5000
max_text_length = 500
embedding_vecor_dimensionality = 32
random_seed = 42
np.random.seed(random_seed)
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=n_top_words,
                                                         skip_top=0,
                                                         maxlen=max_text_length,
                                                         test_split=0.2,
                                                         seed=random_seed)
word_index = reuters.get_word_index(path="reuters_word_index.json")
X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Embedding(n_top_words, embedding_vecor_dimensionality, input_length=max_text_length))
model.add(LSTM(100))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
