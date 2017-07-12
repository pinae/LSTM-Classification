#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import os
import json
from bs4 import BeautifulSoup


def extract_plaintext(html_text):
    soup = BeautifulSoup(html_text)
    lines = soup.get_text().split("\n")
    non_epty_lines = []
    for line in lines:
        if len(line.strip()) > 0:
            non_epty_lines.append(line)
    return "\n\n".join(non_epty_lines)


def extract_data(json_data):
    if not json_data:
        print("ERROR: No data!")
        return None
    if "content" not in json_data:
        return None
    if "keywords" not in json_data or not json_data["keywords"]:
        return None
    if not json_data["content"] or "text_html" not in json_data["content"]:
        print("ERROR: No content!")
        return None
    if "title" not in json_data["content"]:
        print("ERROR: No title.")
        return None
    # print(json.dumps(json_data, indent=2))
    categories = []
    for keyword_dict in json_data["keywords"]:
        if "keyword" in keyword_dict.keys():
            categories.append(keyword_dict["keyword"])
    if len(categories) > 0:
        return {
            "text": extract_plaintext(json_data["content"]["text_html"]),
            "title": json_data["content"]["title"],
            "categories": categories
        }


def read_files(base_path):
    valid_data = []
    for channel in ["autos", "ct", "ct-tv", "developer", "foto", "hardware-hacks", "ix", "mac-and-i",
                    "mobil", "netze", "open", "resale", "security", "tr"]:
        for filename in os.listdir(os.path.join(base_path, channel, "artikel")):
            print("Processing: " + os.path.join(base_path, channel, "artikel", filename))
            with open(os.path.join(base_path, channel, "artikel", filename)) as json_file:
                data = extract_data(json.load(json_file))
                if data:
                    valid_data.append(data)
    for cat in ["meldung", "specials", os.path.join("tp", "news")]:
        for filename in os.listdir(os.path.join(base_path, cat)):
            print("Processing: " + os.path.join(base_path, cat, filename))
            with open(os.path.join(base_path, cat, filename)) as json_file:
                data = extract_data(json.load(json_file))
                if data:
                    valid_data.append(data)
    with open(os.path.join("heise-online-dataset", "heise-online_tagged.json"), 'w') as out_f:
        json.dump(valid_data, out_f)


if __name__ == "__main__":
    read_files("/home/jme/ml-data/heise-online/")
