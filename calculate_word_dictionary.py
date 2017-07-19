#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
import json
import os
import h5py

START_CHARS = "\"'„([{"
END_CHARS = ".,?!;:\"'“}])"


def print_sorted_word_dict(words):
    i = 0
    for key in sorted(words.items(), key=lambda count: count[1], reverse=True):
        if i < 500:
            print(key[0] + ": " + str(key[1]))
        else:
            break
        i += 1


def create_numbered_word_dict(words):
    numbered_word_dict = {}
    for i, key in enumerate(sorted(words.items(), key=lambda count: count[1], reverse=True)):
        numbered_word_dict[key[0]] = i+1
    return numbered_word_dict


def create_word_histogram(text):
    words = {}
    for word in text.split():
        if word[0] in START_CHARS and len(word) > 1:
            s = word[0]
            if s in words:
                words[s] += 1
            else:
                words[s] = 1
            word = word[1:]
        if word[len(word)-1] in END_CHARS and len(word) > 1:
            s = word[len(word)-1]
            if s in words:
                words[s] += 1
            else:
                words[s] = 1
            word = word[:-1]
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    return words


def text_to_int_list(text, words):
    int_text = []
    for word in text.split():
        if word[0] in START_CHARS and len(word) > 1:
            int_text.append(words[word[0]])
            word = word[1:]
        if word[len(word)-1] in END_CHARS and len(word) > 1:
            int_text.append(words[word[:-1]])
            int_text.append(words[word[len(word) - 1]])
        else:
            int_text.append(words[word])
    return int_text


if __name__ == "__main__":
    max_article_length = 0
    max_cat_count = 0
    article_list = []
    with open(os.path.join("heise-online-dataset", "heise-online_tagged.json"), 'r') as f:
        data = json.load(f)
        full_text = ""
        category_counts = {}
        for article in data:
            full_text += article["text"] + "\n\n"
            for c in article["categories"]:
                if c not in category_counts.keys():
                    category_counts[c] = 1
                else:
                    category_counts[c] += 1
        category_list = [i[0] for i in sorted(category_counts.items(), key=lambda count: count[1], reverse=True)[:100]]
        print("Text and categories collected.")
        word_dict = create_word_histogram(full_text)
        print("Word histogram created.")
        numbered_words = create_numbered_word_dict(word_dict)
        print("Numbered the words.")
        for article in data:
            cat_list = []
            for c in article["categories"]:
                if c in category_list:
                    cat_list.append(category_list.index(c) + 1)
            if len(cat_list) > 0:
                int_text = text_to_int_list(article["text"], numbered_words)
                max_article_length = max(max_article_length, len(int_text))
                max_cat_count = max(max_cat_count, len(cat_list))
                article_list.append((int_text, cat_list))
        print("Words converted.")
    with h5py.File(os.path.join("heise-online-dataset", "heise-online.hdf5"), "w") as hdf5_file:
        ho_inputs = hdf5_file.create_dataset("heise-online_texts",
                                             (len(article_list), max_article_length), dtype='i',
                                             compression="gzip", shuffle=True)
        ho_categories = hdf5_file.create_dataset("heise-online_categories",
                                                 (len(article_list), max_cat_count), dtype='i',
                                                 compression="gzip", shuffle=True)
        for i, data_arrays in enumerate(article_list):
            ho_inputs[i] = np.pad(data_arrays[0], (max_article_length-len(data_arrays[0]), 0),
                                  mode='constant', constant_values=0)
            ho_categories[i] = np.pad(data_arrays[1], (max_cat_count-len(data_arrays[1]), 0),
                                      mode='constant', constant_values=data_arrays[1][0])
            if i % 100 == 0:
                print("Written " + str(i) + " of " + str(len(article_list)))
        ho_inputs.attrs['words'] = json.dumps(numbered_words)
        ho_inputs.attrs['word_histogram'] = json.dumps(word_dict)
        ho_categories.attrs['categories'] = json.dumps(category_list)
        print("File written.")
