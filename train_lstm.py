#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
import keras.backend as K
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
#from reuters_dataset import reuters_ingredient, load_data, get_word_list
from heise_online_dataset import heise_online_ingredient, load_data, get_word_count

ex = Experiment('LSTM_Classification', ingredients=[heise_online_ingredient])
ex.observers.append(MongoObserver.create())
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, logs):
    _run.add_artifact("weights.hdf5")
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("binary_accuracy", float(logs.get('binary_accuracy')))
    _run.log_scalar("c_score", float(logs.get('c_score')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("val_binary_accuracy", float(logs.get('val_binary_accuracy')))
    _run.log_scalar("val_c_score", float(logs.get('val_c_score')))
    _run.result = float(logs.get('val_c_score'))


class LogPerformance(Callback):
    def on_epoch_end(self, batch, logs={}):
        log_performance(logs=logs)


def c_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    positive_points = K.sum(y_true*y_pred)
    negative_guesses = K.sum(K.clip(y_pred-y_true, 0.0, 1.0))
    return (positive_points-negative_guesses)/K.sum(y_true)


@ex.config
def my_config():
    embedding_vector_dimensionality = 128
    embedding_dropout_factor = 0.2
    recurrent_dropout_factor = 0.05
    LSTM_dropout_factor = 0.05
    layer_dropout_factor = 0.05
    LSTM_layer_sizes = [100, 100]
    lr = 0.03
    lr_decay = 0.98
    batch_size = 512
    epoch_no = 100
    max_train_size = None  # whole dataset
    max_test_size = None  # whole dataset


@ex.automain
def train_network(embedding_vector_dimensionality, embedding_dropout_factor, recurrent_dropout_factor,
                  LSTM_dropout_factor, layer_dropout_factor, LSTM_layer_sizes, lr, lr_decay,
                  batch_size, epoch_no, max_train_size, max_test_size):
    X_train, y_train, X_test, y_test = load_data()
    if max_train_size and type(max_train_size) == int:
        X_train = X_train[:max_train_size]
        y_train = y_train[:max_train_size]
    if max_test_size and type(max_test_size) == int:
        X_test = X_test[:max_test_size]
        y_test = y_test[:max_test_size]
    print("Shape of the training input: (%d, %d)" % X_train.shape)
    print("Shape of the training output: (%d, %d)" % y_train.shape)
    model = Sequential()
    model.add(Embedding(get_word_count(), embedding_vector_dimensionality, input_length=X_train.shape[1]))
    model.add(Dropout(embedding_dropout_factor))
    for size in LSTM_layer_sizes[:-1]:
        model.add(LSTM(units=size, return_sequences=True,
                       recurrent_dropout=recurrent_dropout_factor,
                       dropout=LSTM_dropout_factor))
        model.add(Dropout(layer_dropout_factor))
    model.add(LSTM(units=LSTM_layer_sizes[-1], recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
    model.add(Dropout(layer_dropout_factor))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    optimizer = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy', c_score])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_no, batch_size=batch_size,
              callbacks=[ModelCheckpoint("weights.hdf5", monitor='val_loss',
                                         save_best_only=True, mode='auto', period=1),
                         LogPerformance()])

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
