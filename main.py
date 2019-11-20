from preprcessing import *
from plot import *
from clustering import *


def plot_cluster_validate_data(df, title_prefix):
    plot_data(df, title_prefix)

    min_pts = 3

    knn_distance_plot(df, min_pts, title_prefix)

    # apply dbscan
    dbscan_labels = apply_dbscan(df, 0.001, min_pts)
    plot_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    # apply dbcv
    # apply_dbcv(df, dbscan_labels)

    # apply optics
    optics_clust = apply_optics(df, 100, 2)
    plot_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

    # apply dbcv
    # apply_dbcv(df, optics_clust.labels_)

    # reachability plot
    # reachability_plot(df, optics_clust)

    # apply cluster_optics_dbscan
    # dbscan_labels = apply_cluster_optics_dbscan(optics_clust, 0.002)
    # plot_clusters(pickup_df, optics_dbscan_labels, title_prefix)


def analyze_hourly_pickup_data(df):
    hourly_pickup_df_list = get_hourly_data(df, 'pickup')

    for hour, hourly_pickup_df in enumerate(hourly_pickup_df_list):
        plot_cluster_validate_data(hourly_pickup_df, 'Pickup at ' + str(hour))
        break


def analyze_hourly_dropoff_data(df):
    hourly_dropoff_df_list = get_hourly_data(df, 'dropoff')

    for hour, hourly_dropoff_df in enumerate(hourly_dropoff_df_list):
        plot_cluster_validate_data(hourly_dropoff_df, 'Dropoff at ' + str(hour))
        break


def analyze_whole_pickup_data(df):

    pickup_df = get_whole_specific_data(df, 'pickup')

    # plot, custer, validate pickup data
    plot_cluster_validate_data(pickup_df, 'Pickup')


def analyze_whole_dropoff_data(df):

    dropoff_df = get_whole_specific_data(df, 'dropoff')

    # plot, custer, validate dropoff data
    plot_cluster_validate_data(dropoff_df, 'Dropoff')


def main():
    # read and preprocess data
    df = read_and_preprocess_data()

    sampled_df = sample_data(df)

    analyze_hourly_pickup_data(df)


if __name__ == "__main__":
    main()
