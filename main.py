import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


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
                     date_parser=parse_date, nrows=100000,
                     dtype={"pickup_longitude": "float64", "pickup_latitude": "float64", "dropoff_longitude": "float64",
                            "dropoff_latitude": "float64"})

    # df.info()
    return df


def knn_distance_plot(df, ns=3):
    nbrs = NearestNeighbors(n_neighbors=ns).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distance_dec = sorted(distances[:, ns - 1], reverse=True)

    # see the difference between linear and log ploting
    # plt.yscale('linear')
    plt.yscale('log')

    plt.plot(list(range(1, len(df) + 1)), distance_dec)
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
    # print(pickup_df.tail(10))

    # pickup_df.plot.scatter(x='pickup_longitude', y='pickup_latitude', s=1, figsize=(8, 6))
    # plt.show()

    dropoff_df = df[['dropoff_longitude', 'dropoff_latitude']]
    # print(dropoff_df.tail(10))

    # dropoff_df.plot.scatter(x='dropoff_longitude', y='dropoff_latitude', s=5, figsize=(8, 6))
    # plt.show()

    # knn-distance plot for pickup, knee point distance = 0.001
    # knn_distance_plot(pickup_df, 3)

    # knn-distance plot for dropoff, knee point distance = 0.005
    knn_distance_plot(dropoff_df, 3)


if __name__ == "__main__":
    main()
