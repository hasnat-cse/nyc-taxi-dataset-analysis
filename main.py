from cluster_plot_validate import *
from plot import *
from preprcessing import *

from enum_classes import ClusteringMethod


def analyze_periodic_data(df, periods, clustering_method, title_prefix):
    periodic_df_list = get_periodic_data(df, periods)

    for i, periodic_df in enumerate(periodic_df_list):
        time_range = str(periods[i][0]) + ' to ' + str(periods[i][1])

        title = title_prefix + " at " + time_range + " with " + str(len(periodic_df)) + " Trips"

        plot_trip_data_on_map(periodic_df, title)

        clust_plot_validate_method = clustering_method_to_functions(clustering_method)

        if clustering_method is not None:
            clust_plot_validate_method(periodic_df, title_prefix + " at " + time_range)


def analyze_data(df, title_prefix):
    df = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]

    title = title_prefix + " with " + str(len(df)) + " Trips"

    plot_trip_data_on_map(df, title)

    cluster_plot_validate_hdbscan(df, title_prefix)


def main():
    # read and preprocess data
    weekday_df, weekend_df = read_and_preprocess_data()

    # analyze_data(weekday_df, "Weekday")
    # analyze_data(weekend_df, "Weekend")

    # periods = [(0, 6), (6, 10), (10, 15), (15, 19), (19, 24)]
    periods = [(10, 15), (15, 19), (19, 24)]

    analyze_periodic_data(weekday_df, periods, ClusteringMethod.hdbscan, "Weekday")
    # analyze_periodic_data(weekend_df, periods, ClusteringMethod.hdbscan, "Weekend")


if __name__ == "__main__":
    main()
