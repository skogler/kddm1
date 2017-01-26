#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

from collections import Counter
from collections import defaultdict

URL_KEY = 'url'
ING_KEY = 'ingredients'
TITLE_KEY = 'title'

def printRecipeSamples(num_samples, labels, n_clusters):
    with open('cache/recipe_names.dat', 'rb') as fdat:
        recipes = pickle.load(fdat)
    for cluster in range(n_clusters):
        print("Printing recipes for cluster: " + str(cluster))
        ids = np.where(labels == cluster)
        counter = 0
        for id in ids[counter]:
            if counter == num_samples:
                break
            print(recipes[id]["title"])
            counter += 1

def printClusterDiffs(cluster_centers, feature_labels):
    for i, cluster1 in enumerate(cluster_centers):
        for j, cluster2 in enumerate(cluster_centers):
            if i >= j:
                continue
            diff = np.absolute(cluster2 - cluster1)
            maxDiff = np.max(diff)
            index = np.argmax(diff)
            print("Max Diff of cluster {} to cluster {} is: {} which is {}".format(i, j, maxDiff, feature_labels[index]))

def printClusterUniqueness(cluster_centers, feature_labels):
    for i, cluster1 in enumerate(cluster_centers):
        clusterMean = np.zeros(cluster1.shape)
        for j, cluster2 in enumerate(cluster_centers):
            clusterMean =clusterMean + cluster2

        diff = np.absolute(clusterMean/9 - cluster1)
        maxDiff = np.max(diff)
        index = np.argmax(diff)
        print("Max Uniqueness of cluster {} is: {} which is {}".format(i, maxDiff, feature_labels[index]))

def visualization(labels, feature_labels):
    clusters = set(labels)
    n_clusters_ = len(clusters)

    if -1 in labels:
        n_clusters_ -= 1

    print('num clusters: {}'.format(n_clusters_))
    print('silhouette: {:0.3f}'.format(metrics.silhouette_score(X, labels)))
    print('entropy of labels: {:0.3f}'.format(metrics.cluster.entropy(labels)))

    num_samples_per_cluster = np.bincount(labels)

    cluster_centers = np.zeros(n_clusters_ * X.shape[1]).reshape(n_clusters_, X.shape[1])

    for i in range(n_clusters_):
        cluster_centers[i, :] = np.average(X[np.where(labels == i), :], 1)

    plt.figure()
    bars = plt.bar(sorted(list(clusters)), num_samples_per_cluster)
    #bars[0].set_color('r')
    plt.title('Clustering of recipes')
    plt.savefig('cluster_bars.png')

    max_labellen = max([len(x) for x in feature_labels])

    for i in range(n_clusters_):
        print('Cluster {} (size {}):'.format(i, num_samples_per_cluster[i]))
        for f in range(X.shape[1]):
            print('  {:{width}} : {:0.4f}'.format(feature_labels[f], cluster_centers[i, f], width=max_labellen))

    printRecipeSamples(5, labels, 10) # immer zu n_clusters anpassen
    printClusterDiffs(cluster_centers, feature_labels)
    printClusterUniqueness(cluster_centers, feature_labels)

def clustering2(X, feature_labels):
    n_clusters = 10

    clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters)

    clustering.fit(X)

    labels = clustering.labels_

    visualization(labels, feature_labels)

def clustering(X, feature_labels):

    print('input dimensions:', X.shape)
    # X = StandardScaler().fit_transform(X)
    clusterer = DBSCAN(eps=1.0/4.0 , metric='jaccard')

    result = clusterer.fit(X)

    core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
    core_samples_mask[result.core_sample_indices_] = True
    labels = result.labels_
    
    visualization(labels, feature_labels)

if __name__ == '__main__':
    filename = 'cache/recipes_X.npz'
    with np.load(filename) as data:
        X = data['X'][:,:]
    with open('cache/features.dat', 'rb') as fdat:
        features = pickle.load(fdat)
    clustering2(X, features)
