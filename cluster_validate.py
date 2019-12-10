import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score

from DBCV import DBCV


def apply_dbscan(df, eps, min_samples):
    ##############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

    return db.labels_


def apply_optics(df, min_samples, max_eps=np.inf):
    ##############################################################################
    # Compute OPTICS
    clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=.05).fit(df)

    return clust


def apply_hdbscan(df, min_cluster_size, min_samples):
    ##############################################################################
    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(df)

    return cluster_labels


def apply_cluster_optics_dbscan(clust, eps):
    labels = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=eps)

    return labels


def apply_dbcv(df, labels):
    np_array = df.to_numpy()
    score = DBCV(np_array, labels, dist_function=euclidean)
    print("DBCV Score: %s" % score)


def calculate_silhouette_score(df, labels):
    np_array = df.to_numpy()
    score = silhouette_score(np_array, labels, metric=euclidean)
    print("Silhouette Score: %s" % score)
