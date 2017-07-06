#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('LSTM_Classification')
ex.observers.append(MongoObserver.create())


@ex.config
def my_config():
    n_top_words = 5000
    max_text_length = 500
    embedding_vecor_dimensionality = 16
    recurrent_dropout_factor = 0.2
    LSTM_dropout_factor = 0.2
    layer_dropout_factor = 0.0


@ex.capture
def get_data(n_top_words, max_text_length, seed):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                             num_words=n_top_words,
                                                             skip_top=0,
                                                             maxlen=max_text_length,
                                                             test_split=0.2,
                                                             seed=seed)
    # word_index = reuters.get_word_index(path="reuters_word_index.json")
    X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test


@ex.automain
def train_network(n_top_words, max_text_length, embedding_vecor_dimensionality,
                  recurrent_dropout_factor, LSTM_dropout_factor, layer_dropout_factor):
    X_train, y_train, X_test, y_test = get_data()
    model = Sequential()
    model.add(Embedding(n_top_words, embedding_vecor_dimensionality, input_length=max_text_length))
    model.add(Dropout(layer_dropout_factor))
    model.add(LSTM(100, recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
    model.add(Dropout(layer_dropout_factor))
    model.add(Dense(46, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
