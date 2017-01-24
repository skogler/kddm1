#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

URL_KEY = 'url'
ING_KEY = 'ingredients'
TITLE_KEY = 'title'


# will also display the count of different ingredients
def apply_threshold_to_counter(counter, threshold):
    counter_thresh = dict(counter)  # apply threshold
    for key, count in counter.items():
        if count < threshold:
            del counter_thresh[key]
    print('threshold of recipes:', threshold, 'different ingredients:', len(counter_thresh.keys()))
    return counter_thresh


# this will build the json the d3js-html then will read
# has a "strange' json structure, don't ask me who thought of that....
# further info: https://d3js.org/ or talk to thorsten for now :-P
def build_bubble_json(used, out):
    # build bubble chart json structure
    bubble_dict = defaultdict(list)
    for key, value in used.items():
        parent_item = {}
        item = {}
        parent_item['children'] = []
        parent_item['name'] = key
        parent_item['children'].append(item)

        item['name'] = key
        item['size'] = value
        item['url'] = ''

        bubble_dict['children'].append(parent_item)

    with open(out, 'wt') as outfile:
        json.dump(bubble_dict, outfile, sort_keys=True)


# sanitize using stopword removal and regex
def sanitize(ingredients_raw):
    stemmed = []
    # stemmer = GermanStemmer()
    # stop words from a german grammar
    stop_words = set(stopwords.words('german'))

    for ing in ingredients_raw:
        # remove braces, replace every other char (e.g. comma) with space
        # why space? picture this: "Salz,Pfeffer" we want this to be "Salz Pfeffer" for our next step
        # the tokenizer doesn't care about two spaces
        san = re.sub(r'\([^)]*\)', '', ing)
        san = re.sub('[^\w]+', ' ', san, flags=re.UNICODE)

        # else:
        for word in san.split():
            if word.lower() not in stop_words and word.istitle():
                stemmed.append(word)
                # else:
                #    print('discarded', word)
    return stemmed


# still working on this one
def calc_zipf(counter):
    all_elems = counter.most_common()  # ('word', 'occurence')
    count = len(all_elems)
    sum20 = 0
    sum80 = 0
    i = 0
    j = 0
    for i in range(0, math.floor(count * 0.05)):
        sum20 = sum20 + (all_elems[i])[1]

    for j in range(math.floor(count * 0.05), count):
        sum80 = sum80 + (all_elems[j])[1]
    print('elems20', i, 'sum20', sum20, 'elems80', j,
          'sum80', sum80)


if __name__ == '__main__':
    recipes_dict = {}
    recipes_san_dict = {}
    ingredient_list = []
    ingredient_san_list = []

    # read data (fairly simple, could be faster - but enough for now)
    print('reading data and sanitizing input (this might take a minute or two)')

    with open('results.jl') as file, open('results_preprocessed.jl', 'wt') as outfile:
        for row in file:
            # load and save non-sanitized ingredients
            recipe = json.loads(row)
            ingredients = recipe[ING_KEY]
            url = recipe[URL_KEY]
            recipes_dict[url] = {TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients}
            ingredient_list.extend(ingredients)

            # ingredient sanitation
            ingredients_preprocessed = sanitize(ingredients)
            recipes_san_dict[url] = {TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients}
            ingredient_san_list.extend(ingredients_preprocessed)

            # now also wright sanitized ingredients to file right away
            recipes_preprocessed = {URL_KEY: url, TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients_preprocessed}
            rec_print = {URL_KEY: url, TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients_preprocessed}
            json.dump(recipes_preprocessed, outfile)
            outfile.write('\n')

    # analyze information
    counter = Counter(ingredient_list)
    calc_zipf(counter)
    print("different ingredients without threshold of recipes it appears in:", len(counter.keys()))
    print('10 most common: ', counter.most_common(20))
    print('now, apply threshold for minimum recipes the ingredients have to occur in')
    apply_threshold_to_counter(counter, 20)
    apply_threshold_to_counter(counter, 50)
    apply_threshold_to_counter(counter, 100)
    used = apply_threshold_to_counter(counter, 1000)

    # first visualization
    build_bubble_json(used, 'bubble.json')

    # analyze information
    overall = 0
    for word, count in counter.most_common():
        overall = overall + count
    print(len(counter.most_common()), 'different ingredients before stem, overall occurrence count', overall)

    counter = Counter(ingredient_san_list)
    calc_zipf(counter)
    print("different ingredients without threshold of recipes it appears in:", len(counter.keys()))
    print('10 most common: ', counter.most_common(20))
    print('now, apply threshold for minimum recipes the ingredients have to occur in')
    apply_threshold_to_counter(counter, 20)
    apply_threshold_to_counter(counter, 50)
    apply_threshold_to_counter(counter, 100)

    # visualize sanitation
    used = apply_threshold_to_counter(counter, 1000)
    build_bubble_json(used, 'bubble_pre.json')
    overall = 0
    for word, count in counter.most_common():
        overall = overall + count

    # analyze information
    print(len(counter.most_common()), 'different ingredients now, overall occurrence count', overall)
