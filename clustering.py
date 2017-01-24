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

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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


# this will build the json the d3js-html then will read
# has a "strange' json structure, don't ask me who thought of that....
# further info: https://d3js.org/ or talk to thorsten for now :-P

def build_bubble_json(used):
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

    with open('bubble.json', 'wt') as outfile:
        json.dump(bubble_dict, outfile, sort_keys=True)


def load_recipes():
    recipes = []
    ingredient_list = []
    # read data (fairly simple, could be faster - but enough for now)

    with open('results.jl') as file:
        # i = 0

        for row in file:
            # i += 1
            # if i > 100000:
            #     break

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

def clustering(X):
    recipes, ingredient_list = load_recipes()
    with open('cache/features.dat', 'rb') as fdat:
        features = pickle.load(fdat)

    print(X.shape)
    # X = StandardScaler().fit_transform(X)

    clusterer = DBSCAN(eps=0.4, metric='cosine', algorithm='brute')

    result = clusterer.fit(X)

    print('fitting finished')
    core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
    core_samples_mask[result.core_sample_indices_] = True
    labels = result.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Silhouette Coefficient: %0.3f'
          % metrics.silhouette_score(X, labels))
    print('Entropy of labels: {}'.format(metrics.cluster.entropy(labels)))

    clusters = set(labels)

    num_samples_per_cluster = np.bincount(labels + 1)

    plt.figure()
    plt.bar(sorted(list(clusters)), num_samples_per_cluster)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        with np.load(filename) as data:
            X = data['X'][:1000,:]
        clustering(X)
    else:
        feature_engineering()
