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

def clustering(X, feature_labels):

    print('input dimensions:', X.shape)
    # X = StandardScaler().fit_transform(X)
    clusterer = DBSCAN(eps=1.0/4.0 , metric='jaccard')

    result = clusterer.fit(X)

    core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
    core_samples_mask[result.core_sample_indices_] = True
    labels = result.labels_
    
    clusters = set(labels)
    n_clusters_ = len(clusters)

    if -1 in labels:
        n_clusters_ -= 1
    
    print('num clusters: {}'.format(n_clusters_))
    print('silhouette: {:0.3f}'.format(metrics.silhouette_score(X, labels)))
    print('entropy of labels: {:0.3f}'.format(metrics.cluster.entropy(labels)))

    num_samples_per_cluster = np.bincount(labels + 1)

    cluster_centers = np.zeros(n_clusters_ * X.shape[1]).reshape(n_clusters_, X.shape[1])

    for i in range(n_clusters_):
        cluster_centers[i,:] = np.average(X[np.where(labels == i),:], 1)

    plt.figure()
    bars = plt.bar(sorted(list(clusters)), num_samples_per_cluster)
    bars[0].set_color('r')
    plt.title('Clustering of recipes')
    plt.savefig('cluster_bars.png')

    max_labellen = max([len(x) for x in feature_labels])

    for i in range(n_clusters_):
        print('Cluster {} (size {}):'.format(i, num_samples_per_cluster[i+1]))
        for f in range(X.shape[1]):
            print('  {:{width}} : {:0.4f}'.format(feature_labels[f], cluster_centers[i,f], width=max_labellen))

if __name__ == '__main__':
    filename = 'cache/recipes_X.npz'
    with np.load(filename) as data:
        X = data['X'][:10000,:]
    with open('cache/features.dat', 'rb') as fdat:
        features = pickle.load(fdat)
    clustering(X, features)
