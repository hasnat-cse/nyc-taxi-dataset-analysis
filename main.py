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

    # apply dbcv
    # apply_dbcv(df, dbscan_labels)

    # calculate silhouette score
    # calculate_silhouette_score(df, dbscan_labels)


def plot_cluster_validate_hdbscan(df, title_prefix):

    min_cluster_size = 10
    min_samples = 3

    # apply hdbscan
    hdbscan_labels = apply_hdbscan(df, min_cluster_size, min_samples)
    plot_clusters(df, hdbscan_labels, title_prefix + ' HDBSCAN')

    # apply dbcv
    # apply_dbcv(df, hdbscan_labels)

    # calculate silhouette score
    # calculate_silhouette_score(df, hdbscan_labels)


def plot_cluster_validate_optics(df, title_prefix):

    min_samples = 100

    # max_eps = 2
    max_eps = float(input("Enter max_eps for OPTICS: "))

    # apply optics
    optics_clust = apply_optics(df, min_samples, max_eps)
    plot_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

    # apply dbcv
    # apply_dbcv(df, optics_clust.labels_)

    # calculate silhouette score
    # calculate_silhouette_score(df, optics_clust.labels_)

    # reachability plot
    # reachability_plot(df, optics_clust)

    # apply cluster_optics_dbscan
    # dbscan_labels = apply_cluster_optics_dbscan(optics_clust, 0.002)
    # plot_clusters(pickup_df, optics_dbscan_labels, title_prefix)


def analyze_hourly_pickup_data(df):
    hourly_pickup_df_list = get_hourly_data(df, 'pickup')

    for hour, hourly_pickup_df in enumerate(hourly_pickup_df_list):
        plot_data(hourly_pickup_df, 'Pickup at ' + str(hour))

        plot_cluster_validate_dbscan(hourly_pickup_df, 'Pickup at ' + str(hour))

        plot_cluster_validate_optics(hourly_pickup_df, 'Pickup at ' + str(hour))

        plot_cluster_validate_hdbscan(hourly_pickup_df, 'Pickup at ' + str(hour))
        break


def analyze_hourly_dropoff_data(df):
    hourly_dropoff_df_list = get_hourly_data(df, 'dropoff')

    for hour, hourly_dropoff_df in enumerate(hourly_dropoff_df_list):
        plot_data(hourly_dropoff_df, 'Dropoff at ' + str(hour))

        plot_cluster_validate_dbscan(hourly_dropoff_df, 'Dropoff at ' + str(hour))

        plot_cluster_validate_optics(hourly_dropoff_df, 'Dropoff at ' + str(hour))

        plot_cluster_validate_hdbscan(hourly_dropoff_df, 'Dropoff at ' + str(hour))
        break


def analyze_whole_pickup_data(df):

    pickup_df = get_whole_specific_data(df, 'pickup')
    plot_data(pickup_df, 'Pickup')

    # plot_cluster_validate_dbscan(pickup_df, 'Pickup')

    plot_cluster_validate_optics(pickup_df, 'Pickup')

    # plot_cluster_validate_hdbscan(pickup_df, 'Pickup')


def analyze_whole_dropoff_data(df):

    dropoff_df = get_whole_specific_data(df, 'dropoff')
    plot_data(dropoff_df, 'Dropoff')

    # plot_cluster_validate_dbscan(dropoff_df, 'Dropoff')

    # plot_cluster_validate_optics(dropoff_df, 'Dropoff')

    plot_cluster_validate_hdbscan(dropoff_df, 'Dropoff')


def main():
    # read and preprocess data
    weekday_df, weekend_df = read_and_preprocess_data()

    analyze_whole_pickup_data(weekday_df)
    analyze_whole_pickup_data(weekend_df)

    # analyze_whole_dropoff_data(weekday_df)
    # analyze_whole_dropoff_data(weekend_df)

    # analyze_hourly_pickup_data(weekday_df)
    # analyze_hourly_pickup_data(weekend_df)

    # analyze_hourly_dropoff_data(weekday_df)
    # analyze_hourly_dropoff_data(weekend_df)


if __name__ == "__main__":
    main()
