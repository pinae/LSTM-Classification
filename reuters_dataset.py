#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Ingredient
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

reuters_ingredient = Ingredient('reuters_dataset')


@reuters_ingredient.config
def cfg():
    n_top_words = 5685
    max_text_length = 500


@reuters_ingredient.capture
def load_data(n_top_words, max_text_length, _seed):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                             num_words=n_top_words,
                                                             skip_top=0,
                                                             maxlen=max_text_length,
                                                             test_split=0.2,
                                                             seed=_seed)
    X_train = sequence.pad_sequences(X_train, maxlen=max_text_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_text_length)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test


@reuters_ingredient.capture
def get_word_list():
    word_index = reuters.get_word_index(path="reuters_word_index.json")
    return word_index


def reconstruct_text(integer_sequence):
    word_index = reuters.get_word_index(path="reuters_word_index.json")
    reversed_index = {v: k for k, v in word_index.items()}
    return " ".join([reversed_index[i] for i in integer_sequence])

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                             num_words=50000,
                                                             skip_top=0,
                                                             maxlen=10000,
                                                             test_split=0.2,
                                                             seed=42)
    print(reconstruct_text(X_train[3]))
