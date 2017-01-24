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

def clustering(X):
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
    filename = 'cache/recipes_X.npz'
    with np.load(filename) as data:
        X = data['X'][:1000,:]
    clustering(X)
