#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
import numpy as np
import os
import h5py
import json

heise_online_ingredient = Experiment('heise_online_dataset')


@heise_online_ingredient.config
def cfg():
    val_split = 0.1
    max_text_length = None
    reduce_dictionary = None


@heise_online_ingredient.capture
def load_data(val_split, max_text_length, reduce_dictionary):
    with h5py.File(os.path.join("heise-online-dataset", "heise-online_old_with_no_zero.hdf5"), 'r') as hdf5_file:
        if max_text_length:
            x = hdf5_file["heise-online_texts"][:, -1*max_text_length:]
        else:
            x = hdf5_file["heise-online_texts"]
        if reduce_dictionary and type(reduce_dictionary) == int:
            x[x >= reduce_dictionary] = 0
        cat_dataset = hdf5_file["heise-online_categories"]
        y = np.sum(np.eye(np.max(cat_dataset) + 1)[cat_dataset], axis=1).clip(max=1)
        combined = np.concatenate((x, y), axis=1)
        np.random.shuffle(combined)
        x, y = np.split(combined, [x.shape[1]], axis=1)
        split = int(x.shape[0]*(1-val_split))
        x_train = x[:split]
        x_test = x[split:]
        y_train = y[:split]
        y_test = y[split:]
        return x_train, y_train, x_test, y_test


@heise_online_ingredient.capture
def get_category_list():
    with h5py.File(os.path.join("heise-online-dataset", "heise-online_old_with_no_zero.hdf5"), 'r') as hdf5_file:
        cat_dataset = hdf5_file["heise-online_categories"]
        category_list = json.loads(cat_dataset.attrs["categories"])
    return category_list


@heise_online_ingredient.capture
def get_word_list():
    with h5py.File(os.path.join("heise-online-dataset", "heise-online_old_with_no_zero.hdf5"), 'r') as hdf5_file:
        input_dataset = hdf5_file["heise-online_texts"]
        word_list = json.loads(input_dataset.attrs['words'])
    return word_list


@heise_online_ingredient.capture
def get_word_count(reduce_dictionary):
    if reduce_dictionary and type(reduce_dictionary) == int:
        return reduce_dictionary
    return len(get_word_list)


@heise_online_ingredient.main
def test_main():
    X_train, y_train, X_test, y_test = load_data()
    print(X_train[42])
    print(y_train[42])

if __name__ == "__main__":
    heise_online_ingredient.run()
