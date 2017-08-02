#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from heise_online_dataset import heise_online_ingredient, load_data, get_word_count
from train_lstm import c_score


ex = Experiment('Dense_Average_Classification', ingredients=[heise_online_ingredient])
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


@ex.config
def my_config():
    embedding_vector_dimensionality = 128
    embedding_dropout_factor = 0.2
    layer_dropout_factor = 0.1
    layer_sizes = [100]
    lr = 0.001
    lr_decay = 0.0
    batch_size = 512
    epoch_no = 150
    max_train_size = None  # whole dataset
    max_test_size = None  # whole dataset


@ex.automain
def train_network(layer_sizes, lr, lr_decay, embedding_vector_dimensionality, embedding_dropout_factor,
                  layer_dropout_factor, batch_size, epoch_no, max_train_size, max_test_size):
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
    for size in layer_sizes:
        model.add(TimeDistributed(Dense(size, activation='relu')))
        model.add(Dropout(layer_dropout_factor))
    model.add(TimeDistributed(Dense(y_train.shape[1], activation='sigmoid')))
    model.add(GlobalAveragePooling1D())
    optimizer = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy', c_score])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_no, batch_size=batch_size,
              callbacks=[ModelCheckpoint("weights.hdf5", monitor='val_loss',
                                         save_best_only=True, mode='auto', period=1),
                         LogPerformance()])

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
