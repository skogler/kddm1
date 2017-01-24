#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.sparse
import scipy.cluster
import scipy.cluster.hierarchy
import pickle

from collections import Counter
from collections import defaultdict

URL_KEY = 'url'
ING_KEY = 'ingredients'
TITLE_KEY = 'title'

# will also display the count of different ingredients
def apply_threshold_to_counter(counter, threshold):
    counter_thresh = dict(counter)
    for key, count in counter.items():
        if count < threshold:
            del counter_thresh[key]

    print('threshold:', threshold, 'different ingredients:', len(counter_thresh.keys()))

    return counter_thresh

def load_recipes():
    recipes = []
    ingredient_list = []

    with open('data/results_preprocessed.jl') as file:
        for row in file:
            recipe = json.loads(row)
            ingredients = recipe[ING_KEY]
            recipes.append({TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients})
            ingredient_list.extend(ingredients)

    return recipes, ingredient_list


def feature_engineering():
    # analyze information

    recipes, ingredient_list = load_recipes()

    counter = Counter(ingredient_list)
    print("different ingredients without threshold of recipes it appears in:", len(counter.keys()))
    print('10 most common: ', counter.most_common(10))
    print('now, apply threshold for minimum recipes the ingredients have to occur in')
    apply_threshold_to_counter(counter, 20)
    apply_threshold_to_counter(counter, 50)
    apply_threshold_to_counter(counter, 100)
    used = apply_threshold_to_counter(counter, 1000)

    features = sorted(used.keys())

    val = [1 if f in recipe[ING_KEY] else 0 
            for recipe in recipes for f in features ]

    X = np.array(val).reshape(len(recipes), len(features))


    np.savez('cache/recipes_X', X=X)
    with open('cache/features.dat', 'wb') as fdat:
        pickle.dump(features, fdat)

if __name__ == '__main__':
    feature_engineering()
