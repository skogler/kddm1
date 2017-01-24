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
    print('threshold:', threshold, 'different ingredients:', len(counter_thresh.keys()))
    return counter_thresh


# can be used to show a histogram of the data
# TODO: dont display x axis
def show_histogram(counter, start=0, to=5000):
    labels, values = zip(*(counter.most_common(to)[start:]))

    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5)
    print('show histogram elements from', start, 'to', to, 'of most occuring ingredients')
    plt.show()


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
    recipes = []
    ingredient_list = []
    ing_counter = Counter()
    with open('results_preprocessed.jl', 'rt') as file:
        for row in file:
            # load and save non-sanitized ingredients
            recipe = json.loads(row)
            recipes.append(recipe)
            ing_counter.update(recipe[ING_KEY])

    show_histogram(ing_counter)
    show_histogram(ing_counter, start=500, to=5500)
    show_histogram(ing_counter, start=1000, to=10000)
    print(ing_counter.most_common(50))
