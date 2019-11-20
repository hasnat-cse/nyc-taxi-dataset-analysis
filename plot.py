import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np


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

    plt.scatter(df['longitude'], df['latitude'], c=df['label'].map(label_color_dict), s=1)

    plt.show()


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


def plot_data(df, title_prefix):
    df.plot.scatter(x='longitude', y='latitude', s=0.5, figsize=(20, 10))

    plt.title(title_prefix + ' Data Points Plot')
    plt.xlabel('Longitiude')
    plt.ylabel('Latitude')

    plt.show()
