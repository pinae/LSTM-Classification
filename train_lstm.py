#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
import keras.backend as K
from sacred import Experiment
from sacred.observers import MongoObserver
#from reuters_dataset import reuters_ingredient, load_data, get_word_list
from heise_online_dataset import heise_online_ingredient, load_data, get_word_count

ex = Experiment('LSTM_Classification', ingredients=[heise_online_ingredient])
ex.observers.append(MongoObserver.create())


def c_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    positive_points = K.sum(y_true*y_pred)
    negative_guesses = K.sum(K.clip(y_pred-y_true, 0.0, 1.0))
    return (positive_points-negative_guesses)/K.sum(y_true)


@ex.config
def my_config():
    embedding_vector_dimensionality = 128
    recurrent_dropout_factor = 0.2
    LSTM_dropout_factor = 0.2
    layer_dropout_factor = 0.0


@ex.automain
def train_network(embedding_vector_dimensionality, recurrent_dropout_factor, LSTM_dropout_factor, layer_dropout_factor):
    X_train, y_train, X_test, y_test = load_data()
    print("Shape of the training input: (%d, %d)" % X_train.shape)
    print("Shape of the training output: (%d, %d)" % y_train.shape)
    model = Sequential()
    model.add(Embedding(get_word_count(), embedding_vector_dimensionality, input_length=X_train.shape[1]))
    model.add(Dropout(layer_dropout_factor))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
    model.add(Dropout(layer_dropout_factor))
    model.add(LSTM(128, recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
    model.add(Dropout(layer_dropout_factor))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', c_score])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
