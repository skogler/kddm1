#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import math
import matplotlib.pyplot as plt
import collections
import numpy as np
import scipy.sparse
import scipy.cluster
import scipy.cluster.hierarchy
import pickle
import random

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

    print('threshold > {} recipes results in {} features'.format(threshold, len(counter_thresh.keys())))

    return counter_thresh

def load_recipes():
    recipes = []
    ingredient_list = []

    with open('data/results_preprocessed.jl') as file:
        for row in file:
            recipe = json.loads(row)
            ingredients = list(set(recipe[ING_KEY]))
            recipes.append({TITLE_KEY: recipe[TITLE_KEY], ING_KEY: ingredients})
            ingredient_list.extend(ingredients)

    return recipes, ingredient_list

def joint_probability(x, y, cooccurrence, num_recipes):
    res = cooccurrence[x,y] / num_recipes
    assert res >= 0 and res <= 1
    return res

def conditional_probability(y, x, cooccurrence):
    res = cooccurrence[x,y] / cooccurrence[x,x]
    assert res >= 0 and res <= 1
    return res

def conditional_entropy(y, X, cooccurrence, num_recipes):
    ent = 0
    for x in X:
        cond = conditional_probability(y, x, cooccurrence)
        joint = joint_probability(x, y, cooccurrence, num_recipes)
        if cond == 0:
            continue
        ent -=  joint * math.log2(cond)
    assert ent >= 0
    return ent

def entropy(x, cooccurrence, num_recipes):
    res = cooccurrence[x,x] / num_recipes
    res = -res * math.log2(res)
    assert res >= 0
    return res 

def relative_information_gain(y, X, cooccurrence, num_recipes):
    enty = entropy(y, cooccurrence, num_recipes)
    res = (enty - conditional_entropy(y, X, cooccurrence, num_recipes)) / enty
    return res

def feature_selection(recipes, ingredient_list):

    counter = Counter(ingredient_list)

    print("num ingredients:", len(counter.keys()))
    print("num recipes:", len(recipes))

    # APPLY THRESHOLD
    num_recipes = len(recipes)
    low_threshold = 0.15
    used = apply_threshold_to_counter(counter,  int(low_threshold * num_recipes))

    # BUILD CO-OCCURENCE MATRIX
    ingredients_per_recipe = [r[ING_KEY] for r in recipes]
    features = [(i, label, count) for i,(label, count) in enumerate(sorted(used.items(), key=lambda i: i[1], reverse=True))]
    num_features = len(features)

    X = np.array([1 if f in r else 0 
            for r in ingredients_per_recipe for _,f,_ in features]).reshape(len(recipes), len(features))
    cooccurrence = np.dot(np.transpose(X), X)

    # CALCULATE ENTROPY
    probabilities = [(i,label, count / num_recipes) for i,label,count in features]
    rig = [(i,label, -prob * math.log(prob, 2)) for i,label,prob in probabilities]
    rig.sort(key=lambda i: i[2], reverse=True)

    # print('entropy: ', rig)
    # print('ent(s)', entropy(0, cooccurrence, num_recipes))
    # print('ent(p)', entropy(1, cooccurrence, num_recipes))
    # print('cond prob. (s|p)', conditional_probability(0, 1, cooccurrence))
    # print('cond prob. (p|s)', conditional_probability(1, 0, cooccurrence))
    # print('joint prob. (s,p)', joint_probability(0, 1, cooccurrence, num_recipes))
    # print('cond. entr(s, p)', conditional_entropy(0, [1], cooccurrence, num_recipes))
    # print('cond. entr(p, s)', conditional_entropy(1, [0], cooccurrence, num_recipes))

    # SELECT FEATURES BASED ON RELATIVE INFORMATION GAIN

    selection = []
    selection.append(rig.pop(0)[0])

    while len(rig) > 0:
        rig = [(yi, ylabel, relative_information_gain(yi, selection, cooccurrence, num_recipes)) for yi, ylabel, y in rig] 
        rig.sort(key=lambda i: i[2], reverse=True)
        if rig[0][2] <= 0:
            break
        selection.append(rig.pop(0)[0])

    X = X[:, selection]
    feature_labels = [label for i, label,_ in features if i in selection]

    print(X.shape)
    print(feature_labels)

    return X, feature_labels


def feature_engineering():
    recipes, ingredient_list = load_recipes()

    rand_smpl = [recipes[i] for i in sorted(random.sample(range(len(recipes)), 10000))]
    # print(rand_smpl)

    X, feature_labels = feature_selection(rand_smpl, ingredient_list)

    np.savez('cache/recipes_X', X=X)
    with open('cache/features.dat', 'wb') as fdat:
        pickle.dump(feature_labels, fdat)
    with open('cache/recipe_names.dat', 'wb') as fdat:
        pickle.dump(rand_smpl, fdat)

if __name__ == '__main__':
    feature_engineering()
