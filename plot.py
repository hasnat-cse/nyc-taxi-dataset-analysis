from collections import Counter

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

import plotly.graph_objects as go

from enum_classes import LocationType

mapbox_access_token = open(".mapbox_token").read()


def knn_distance_plot(df, ns, title_prefix):
    nbrs = NearestNeighbors(n_neighbors=ns).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distance_dec = sorted(distances[:, ns - 1], reverse=True)

    plt.figure(figsize=(20, 10))
    plt.title(title_prefix + ' K-NN Distance Plot')
    plt.xlabel('Data Points')
    plt.ylabel(str(ns) + '-NN Distance')

    # see the difference between linear and log ploting
    # plt.yscale('linear')
    # plt.yscale('log')

    # plt.plot(list(range(1, len(df) + 1)), distance_dec)
    plt.scatter(list(range(1, len(df) + 1)), distance_dec, s=0.5)
    plt.show()


def plot_clusters(df, labels, title_prefix):
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

    plt.figure(figsize=(20, 10))
    plt.title(title_prefix + ' Clustering Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.scatter(df['dropoff_longitude'], df['dropoff_latitude'], c=df['label'].map(label_color_dict), s=1)

    plt.show()


def plot_topmost_clusters(df, labels, title_prefix):
    df['label'] = labels

    topmost_labels_counter = Counter(x for x in labels if x != -1).most_common()

    topmost_labels = []
    for i, label_counter in enumerate(topmost_labels_counter):
        if i == 3:
            break
        topmost_labels.append(label_counter[0])
        print('Top %s cluster with %s points' % (i + 1, label_counter[1]))

    plt.figure(figsize=(20, 10))
    plt.title(title_prefix + ' Topmost Clustering Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    for label in topmost_labels:
        topmost_df = df[df['label'] == label]
        plt.scatter(topmost_df['pickup_longitude'], topmost_df['pickup_latitude'], c="red", s=1)
        plt.scatter(topmost_df['dropoff_longitude'], topmost_df['dropoff_latitude'], c="blue", s=1)

    plt.show()


def plot_topmost_clusters_on_map(df, labels, title_prefix):
    df['label'] = labels

    topmost_labels_counter = Counter(x for x in labels if x != -1).most_common()

    topmost_labels = []
    for i, label_counter in enumerate(topmost_labels_counter):
        if i == 7:
            break
        topmost_labels.append(label_counter[0])

    for i, label in enumerate(topmost_labels):
        topmost_df = df[df['label'] == label]
        title = title_prefix + " Cluster " + str(i + 1) + " with " + str(len(topmost_df)) + " Trips"
        plot_trip_data_on_map(topmost_df, title)


def reachability_plot(df, clust):
    space = np.arange(len(df))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(20, 10))

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for klass, color in zip(range(0, len(colors)), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        plt.plot(Xk, Rk, color, alpha=0.3)
    plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    # plt.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    # plt.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    plt.show()


def plot_trip_data_on_map(df, title):
    pickup_latitudes = df['pickup_latitude']
    pickup_longitudes = df['pickup_longitude']
    dropoff_latitudes = df['dropoff_latitude']
    dropoff_longitudes = df['dropoff_longitude']

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        name="Pickup",
        lat=pickup_latitudes,
        lon=pickup_longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=2.5,
            symbol='circle',
            color='blue',
            opacity=.8
        )
    ))

    fig.add_trace(go.Scattermapbox(
        name="Dropoff",
        lat=dropoff_latitudes,
        lon=dropoff_longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=2.5,
            symbol='circle',
            color='red',
            opacity=.8
        )
    ))

    fig.update_layout(
        title=title,
        autosize=True,
        hovermode='closest',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=40.7213,
                lon=-73.9871
            ),
            pitch=40,
            zoom=10,
            style="mapbox://styles/hasnat-cse/ck3qt03hc095n1cmbvpr62mo5"
        ),
    )

    fig.show()


def plot_data_points_on_map(latitudes, longitudes, location_type, title):
    if location_type is LocationType.Pickup:
        color = "blue"
    else:
        color = "red"

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        name=location_type.name,
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=2.5,
            symbol='circle',
            color=color,
            opacity=.8
        )
    ))

    fig.update_layout(
        title=title,
        autosize=True,
        hovermode='closest',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=40.7213,
                lon=-73.9871
            ),
            pitch=35,
            zoom=9.5,
            style="mapbox://styles/hasnat-cse/ck3qt03hc095n1cmbvpr62mo5"
        ),
    )

    fig.show()
