import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_date(date_string):
    return pd.datetime.strptime(date_string, "%Y-%d-%m %H:%M:%S")


def remove_rows_that_contain_0_values(df):
    return df[(df['pickup_longitude'] != float(0)) & (df['pickup_latitude'] != float(0)) &
              (df['dropoff_longitude'] != float(0)) & (df['dropoff_latitude'] != float(0))]


def remove_noisy_rows(df):
    return df[(df['pickup_longitude'] < float(-70)) & (df['pickup_latitude'] > float(40)) &
              (df['dropoff_longitude'] < float(-70)) & (df['dropoff_latitude'] > float(40))]


def read_relevant_data():
    df = pd.read_csv("../697_data/yellow_tripdata_2015-09.csv", header=0, usecols=["tpep_pickup_datetime",
                                                                            "tpep_dropoff_datetime", "pickup_longitude",
                                                                            "pickup_latitude", "dropoff_longitude",
                                                                            "dropoff_latitude"],
                     parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
                     date_parser=parse_date, nrows=50000,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})

    # df.info()
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

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

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


def plot_data(df, scale):
    df.plot.scatter(x='longitude', y='latitude', s=scale, figsize=(8, 6))
    plt.show()


def main():
    df = read_relevant_data()
    # print(df.head(10))

    df = remove_rows_that_contain_0_values(df)
    # print(df.tail(10))

    # from the x y plot we see there are a few data points with abnormal longitude or latitude values
    # try commenting following line and see the difference in plot of dropoff
    df = remove_noisy_rows(df)

    pickup_df = df[['pickup_longitude', 'pickup_latitude']]
    # changing column names
    pickup_df = pickup_df.rename(columns={"pickup_longitude": "longitude", "pickup_latitude": "latitude"})
    # print(pickup_df.tail(10))

    # plot pickup data
    plot_data(pickup_df, 1)


    dropoff_df = df[['dropoff_longitude', 'dropoff_latitude']]
    # changing column names
    dropoff_df = dropoff_df.rename(columns={"dropoff_longitude": "longitude", "dropoff_latitude": "latitude"})
    # print(dropoff_df.tail(10))

    # plot dropoff data
    plot_data(dropoff_df, 5)

    # knn-distance plot for pickup, knee point distance = 0.001
    # knn_distance_plot(pickup_df, 3)

    # knn-distance plot for dropoff, knee point distance = 0.005
    # knn_distance_plot(dropoff_df, 3)

    # apply dbscan on pickup
    apply_dbscan(pickup_df, 0.001, 3)

    # apply dbscan on dropoff
    apply_dbscan(dropoff_df, 0.005, 3)


if __name__ == "__main__":
    main()
