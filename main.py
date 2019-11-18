import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors

from DBCV import DBCV
from scipy.spatial.distance import euclidean


def parse_date(date_string):
    return pd.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")


def read_relevant_data():
    df = pd.read_csv("../697_data/yellow_tripdata_2015-09.csv", header=0, usecols=["tpep_pickup_datetime",
                                                                                   "tpep_dropoff_datetime",
                                                                                   "pickup_longitude",
                                                                                   "pickup_latitude",
                                                                                   "dropoff_longitude",
                                                                                   "dropoff_latitude"],
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                     date_parser=parse_date, nrows=351276,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})

    # df.info()
    return df


def remove_rows_that_contain_0_values(df):
    return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
              (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0))]


def remove_noisy_rows(df):
    return df[(df['pickup_longitude'] < float(-70)) & (df['pickup_latitude'] > float(40)) &
              (df['dropoff_longitude'] < float(-70)) & (df['dropoff_latitude'] > float(40))]


def impose_boundary(df):
    return df[(df['pickup_longitude'] <= float(-73.4)) & (df['pickup_longitude'] >= float(-74.4)) &
              (df['pickup_latitude'] >= float(40.5)) & (df['pickup_latitude'] <= float(41)) &
              (df['dropoff_longitude'] <= float(-73.4)) & (df['dropoff_longitude'] >= float(-74.4)) &
              (df['dropoff_latitude'] >= float(40.5)) & (df['dropoff_latitude'] <= float(41))]


def sample_data(df):
    sample_size = 100000
    if len(df) > sample_size:
        df = df.sample(sample_size)

    print(len(df))

    return df


def read_and_preprocess_data():
    df = read_relevant_data()
    # print(df.head(10))

    df = remove_rows_that_contain_0_values(df)
    # print(df.tail(10))

    # from the x y plot we see there are a few data points with abnormal longitude or latitude values
    # try commenting following line and see the difference in plot of dropoff
    df = remove_noisy_rows(df)

    df = impose_boundary(df)

    df = sample_data(df)

    return df


def knn_distance_plot(df, ns):
    nbrs = NearestNeighbors(n_neighbors=ns).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distance_dec = sorted(distances[:, ns - 1], reverse=True)

    # see the difference between linear and log ploting
    # plt.yscale('linear')
    plt.yscale('log')

    plt.plot(list(range(1, len(df) + 1)), distance_dec)
    plt.show()


def apply_dbscan(df, eps, min_samples):
    ##############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

    return db.labels_


def apply_optics(df, min_samples, max_eps=np.inf):
    ##############################################################################
    # Compute OPTICS
    clust = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=.05, min_cluster_size=.05).fit(df)

    return clust


def apply_cluster_optics_dbscan(clust, eps):
    labels = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=eps)

    return labels


def plot_clusters(df, labels):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    df['label'] = labels

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    label_color_dict = {}
    for label, col in zip(unique_labels, colors):

        if label == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        label_color_dict[label] = col

    plt.scatter(df['longitude'], df['latitude'], c=df['label'].map(label_color_dict), s=1)
    plt.show()


def apply_dbcv(df, labels):
    np_array = df.to_numpy()
    score = DBCV(np_array, labels, dist_function=euclidean)
    print("DBCV Score: %s" % score)


def plot_data(df, scale):
    df.plot.scatter(x='longitude', y='latitude', s=scale, figsize=(8, 6))
    plt.show()


def main():
    # read and preprocess data
    df = read_and_preprocess_data()

    pickup_df = df[['pickup_longitude', 'pickup_latitude']]
    # changing column names
    pickup_df = pickup_df.rename(columns={"pickup_longitude": "longitude", "pickup_latitude": "latitude"})
    # print(pickup_df.tail(10))

    # plot pickup data
    plot_data(pickup_df, .5)

    dropoff_df = df[['dropoff_longitude', 'dropoff_latitude']]
    # changing column names
    dropoff_df = dropoff_df.rename(columns={"dropoff_longitude": "longitude", "dropoff_latitude": "latitude"})
    # print(dropoff_df.tail(10))

    # plot dropoff data
    plot_data(dropoff_df, .5)

    # knn-distance plot for pickup, knee point distance = 0.001
    knn_distance_plot(pickup_df, 50)

    # knn-distance plot for dropoff, knee point distance = 0.005
    # knn_distance_plot(dropoff_df, 3)

    # apply dbscan on pickup
    pickup_labels = apply_dbscan(pickup_df, 0.002, 50)
    plot_clusters(pickup_df, pickup_labels)

    # apply dbscan on dropoff
    # dropoff_labels = apply_dbscan(dropoff_df, 0.005, 3)

    # apply dbcv on pickup
    # apply_dbcv(pickup_df, pickup_labels)

    # apply optics
    # pickup_optics_clust = apply_optics(pickup_df, 50, 0.05)

    # apply cluster_optics_dbscan
    # pickup_optics_dbscan_labels = apply_cluster_optics_dbscan(pickup_optics_clust, 0.002)
    # plot_clusters(pickup_df, pickup_optics_dbscan_labels)

    # apply dbcv on pickup
    # apply_dbcv(pickup_df, pickup_optics_dbscan_labels)


if __name__ == "__main__":
    main()
