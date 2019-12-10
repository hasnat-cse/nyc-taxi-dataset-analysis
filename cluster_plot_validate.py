from cluster_validate import *
from plot import knn_distance_plot, plot_clusters, plot_topmost_clusters_on_map
from enum_classes import ClusteringMethod


def clustering_method_to_functions(method):
    switcher = {
        ClusteringMethod.dbscan: cluster_plot_validate_dbscan,
        ClusteringMethod.hdbscan: cluster_plot_validate_hdbscan,
        ClusteringMethod.optics: cluster_plot_validate_optics,
        ClusteringMethod.dbscan_hdbscan: cluster_plot_validate_dbscan_hdbscan,
        ClusteringMethod.optics_hdbscan: cluster_plot_validate_optics_hdbscan,
        ClusteringMethod.all: cluster_plot_validate_all,
    }

    return switcher.get(method, None)


def cluster_plot_validate_dbscan(df, title_prefix):
    min_pts = 7

    knn_distance_plot(df, min_pts, title_prefix)

    # eps = 0.001
    eps = float(input("Enter eps for DBSCAN: "))

    # apply dbscan
    dbscan_labels = apply_dbscan(df, eps, min_pts)

    plot_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    plot_topmost_clusters_on_map(df, dbscan_labels, title_prefix + ' DBSCAN')

    calculate_clustering_scores(df, dbscan_labels)


def cluster_plot_validate_hdbscan(df, title_prefix):
    min_cluster_size = 100
    min_samples = 7

    # apply hdbscan
    hdbscan_labels = apply_hdbscan(df, min_cluster_size, min_samples)

    plot_clusters(df, hdbscan_labels, title_prefix + ' HDBSCAN')

    plot_topmost_clusters_on_map(df, hdbscan_labels, title_prefix + " HDBSCAN")

    calculate_clustering_scores(df, hdbscan_labels)


def cluster_plot_validate_optics(df, title_prefix):
    min_cluster_size = 100
    min_samples = 7

    knn_distance_plot(df, min_samples, title_prefix)

    # max_eps = 5
    max_eps = float(input("Enter max eps for OPTICS: "))

    # apply optics
    optics_clust = apply_optics(df, min_samples, min_cluster_size, max_eps)

    plot_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

    plot_topmost_clusters_on_map(df, optics_clust.labels_, title_prefix + ' OPTICS')

    calculate_clustering_scores(df, optics_clust.labels_)


def cluster_plot_validate_dbscan_hdbscan(df, title_prefix):
    cluster_plot_validate_dbscan(df, title_prefix)
    cluster_plot_validate_hdbscan(df, title_prefix)


def cluster_plot_validate_optics_hdbscan(df, title_prefix):
    cluster_plot_validate_optics(df, title_prefix)
    cluster_plot_validate_hdbscan(df, title_prefix)


def cluster_plot_validate_all(df, title_prefix):
    cluster_plot_validate_dbscan(df, title_prefix)
    cluster_plot_validate_hdbscan(df, title_prefix)
    cluster_plot_validate_optics(df, title_prefix)


def calculate_clustering_scores(df, labels):
    # calculate validation scores
    sil_score = calculate_silhouette_score(df, labels, 1000, 5)
    print("Approximate Silhouette Score: %s" % sil_score)

    ch_score = calculate_calinski_harabasz_score(df, labels)
    print("Calinski Harabasz Score: %s" % ch_score)

    db_score = calculate_davies_bouldin_score(df, labels)
    print("Davies Bouldin Score: %s" % db_score)