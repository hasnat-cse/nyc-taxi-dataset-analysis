from preprcessing import *
from plot import *
from clustering import *


def plot_cluster_validate_dbscan(df, title_prefix):

    min_pts = 3

    knn_distance_plot(df, min_pts, title_prefix)

    # eps = 0.001
    eps = float(input("Enter eps for DBSCAN: "))

    # apply dbscan
    dbscan_labels = apply_dbscan(df, eps, min_pts)
    plot_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    plot_topmost_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    # apply dbcv
    # apply_dbcv(df, dbscan_labels)

    # calculate silhouette score
    # calculate_silhouette_score(df, dbscan_labels)


def plot_cluster_validate_hdbscan(df, title_prefix):

    min_cluster_size = 100
    min_samples = 3

    # apply hdbscan
    hdbscan_labels = apply_hdbscan(df, min_cluster_size, min_samples)
    plot_clusters(df, hdbscan_labels, title_prefix + ' HDBSCAN')

    plot_topmost_clusters(df, hdbscan_labels, title_prefix + ' HDBSCAN')

    # apply dbcv
    # apply_dbcv(df, hdbscan_labels)

    # calculate silhouette score
    calculate_silhouette_score(df, hdbscan_labels)


def plot_cluster_validate_optics(df, title_prefix):

    min_samples = 50

    max_eps = 2

    # apply optics
    optics_clust = apply_optics(df, min_samples, max_eps)
    plot_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

    plot_topmost_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

    # apply dbcv
    # apply_dbcv(df, optics_clust.labels_)

    # calculate silhouette score
    # calculate_silhouette_score(df, optics_clust.labels_)

    # reachability plot
    # reachability_plot(df, optics_clust)

    # apply cluster_optics_dbscan
    # dbscan_labels = apply_cluster_optics_dbscan(optics_clust, 0.002)
    # plot_clusters(pickup_df, optics_dbscan_labels, title_prefix)


def analyze_periodic_data(df, data_type):
    # periods = [(0, 6), (6, 10), (10, 15), (15, 19), (19, 24)]
    periods = [(6, 10)]

    periodic_df_list = get_periodic_data(df, periods, data_type)

    for i, periodic_df in enumerate(periodic_df_list):
        if data_type == 'pickup':
            title_prefix = 'Pickup at ' + str(periods[i][0]) + ' to ' + str(periods[i][1])

        elif data_type == 'dropoff':
            title_prefix = 'Dropoff at ' + str(periods[i][0]) + ' to ' + str(periods[i][1])

        else:
            return

        plot_data(periodic_df, title_prefix)

        plot_cluster_validate_dbscan(periodic_df, title_prefix)

        plot_cluster_validate_optics(periodic_df, title_prefix)

        plot_cluster_validate_hdbscan(periodic_df, title_prefix)


def analyze_whole_data(df, data_type):

    df = get_whole_specific_data(df, data_type)

    if data_type == 'pickup':
        title_prefix = 'Pickup'

    elif data_type == 'dropoff':
        title_prefix = 'Dropoff'

    else:
        return

    plot_data(df, title_prefix)

    plot_cluster_validate_dbscan(df, title_prefix)

    plot_cluster_validate_optics(df, title_prefix)

    plot_cluster_validate_hdbscan(df, title_prefix)


def main():
    # read and preprocess data
    weekday_df, weekend_df = read_and_preprocess_data()

    # analyze_whole_data(weekday_df, 'pickup')
    # analyze_whole_data(weekend_df, 'pickup')

    # analyze_whole_data(weekday_df, 'dropoff')
    # analyze_whole_data(weekend_df, 'dropoff')

    # analyze_periodic_data(weekday_df, 'pickup')
    analyze_periodic_data(weekend_df, 'pickup')

    # analyze_periodic_data(weekday_df, 'dropoff')
    # analyze_periodic_data(weekend_df, 'dropoff')


if __name__ == "__main__":
    main()
