from preprcessing import *
from plot import *
from clustering import *


def plot_cluster_validate_dbscan(df, title_prefix):

    min_pts = 9

    knn_distance_plot(df, min_pts, title_prefix)

    # eps = 0.001
    eps = float(input("Enter eps for DBSCAN: "))

    # apply dbscan
    dbscan_labels = apply_dbscan(df, eps, min_pts)

    # plot_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    plot_topmost_clusters(df, dbscan_labels, title_prefix + ' DBSCAN')

    # apply dbcv
    # apply_dbcv(df, dbscan_labels)

    # calculate silhouette score
    # calculate_silhouette_score(df, dbscan_labels)


def plot_cluster_validate_hdbscan(df, title_prefix):

    min_cluster_size = 100
    min_samples = 7

    # apply hdbscan
    hdbscan_labels = apply_hdbscan(df, min_cluster_size, min_samples)

    # plot_clusters(df, hdbscan_labels, title_prefix + ' HDBSCAN')

    plot_topmost_clusters_on_map(df, hdbscan_labels, title_prefix + " HDBSCAN")

    # apply dbcv
    # apply_dbcv(df, hdbscan_labels)

    # calculate silhouette score
    # calculate_silhouette_score(df, hdbscan_labels)


def plot_cluster_validate_optics(df, title_prefix):

    min_samples = 7

    knn_distance_plot(df, min_samples, title_prefix)

    # max_eps = 5
    max_eps = float(input("Enter max eps for OPTICS: "))

    # apply optics
    optics_clust = apply_optics(df, min_samples, max_eps)

    # plot_clusters(df, optics_clust.labels_, title_prefix + ' OPTICS')

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


def analyze_periodic_data(df, title_prefix):
    # periods = [(0, 6), (6, 10), (10, 15), (15, 19), (19, 24)]
    periods = [(15, 19)]

    periodic_df_list = get_periodic_data(df, periods)

    for i, periodic_df in enumerate(periodic_df_list):
        time_range = str(periods[i][0]) + ' to ' + str(periods[i][1])

        title_pickup = title_prefix + " Pickup at " + time_range + " with " + str(len(periodic_df)) + " Trips"
        title_dropoff = title_prefix + " Dropoff at " + time_range + " with " + str(len(periodic_df)) + " Trips"

        plot_data_points_on_map(periodic_df['pickup_latitude'], periodic_df['pickup_longitude'], "Pickup", title_pickup)
        plot_data_points_on_map(periodic_df['dropoff_latitude'], periodic_df['dropoff_longitude'], "Dropoff", title_dropoff)

        # plot_cluster_validate_dbscan(periodic_df, title_prefix)

        # plot_cluster_validate_optics(periodic_df, title_prefix)

        plot_cluster_validate_hdbscan(periodic_df, title_prefix + " at " + time_range)


def analyze_data(df, title_prefix):
    df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]

    title_pickup = title_prefix + " Pickup with " + str(len(df)) + " Trips"
    title_dropoff = title_prefix + " Dropoff with " + str(len(df)) + " Trips"

    plot_data_points_on_map(df['pickup_latitude'], df['pickup_longitude'], "Pickup", title_pickup)
    plot_data_points_on_map(df['dropoff_latitude'], df['dropoff_longitude'], "Dropoff", title_dropoff)

    # plot_cluster_validate_dbscan(periodic_df, title_prefix)

    # plot_cluster_validate_optics(periodic_df, title_prefix)

    plot_cluster_validate_hdbscan(df, title_prefix)


def main():
    # read and preprocess data
    weekday_df, weekend_df = read_and_preprocess_data()

    analyze_data(weekday_df, "Weekday")
    analyze_data(weekend_df, "Weekday")

    # analyze_periodic_data(weekday_df, "Weekday")
    # analyze_periodic_data(weekend_df, "Weekend")


if __name__ == "__main__":
    main()
