import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data_set = pd.read_csv('variant_1.csv')
data_frame = pd.DataFrame(data_set)

X1 = data_frame.values
pca = PCA(n_components=2)
fit = pca.fit(X1)
features = fit.transform(X1)


def clustering(algorythm, feat, title: str):
    label = algorythm.fit_predict(feat)
    label0 = feat[label == 0]
    label1 = feat[label == 1]
    label2 = feat[label == 2]
    plt.scatter(label0[:, 0], label0[:, 1], s=5, color='blue')
    plt.scatter(label1[:, 0], label1[:, 1], s=10, color='black')
    plt.scatter(label2[:, 0], label2[:, 1], s=5, color='g')
    plt.title(title)
    plt.show()
    return label

# Kmeans

def dbscan_clustering(feat):
    def __visualizing_results(labels, X, core_samples_mask, n_clusters_):
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("Estimated number of clusters: %d" % n_clusters_)
        plt.show()

    dbscan = DBSCAN(eps=0.1, min_samples=8).fit(feat)
    labels = dbscan.labels_
    core_samples_mask = [labels.core_sample_indices_] = True

    # Number of Clusters
    N_clus = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated no. of clusters: %d' % N_clus)
    __visualizing_results(labels, X1, core_samples_mask, N_clus)


K_means = KMeans(n_clusters=3)
spectral = SpectralClustering(n_clusters=3)
agglomerative = AgglomerativeClustering(n_clusters=3)

algorythms = [
    K_means,
    spectral,
    agglomerative
]
titles = [
    'KMeans Clustering',
    'Spectral Clustering',
    'Agglomerative Clustering'
]
labels = []

for i in range(len(algorythms)):
    label = clustering(algorythms[i], features, titles[i])
    labels.append(label)

for i in range(len(labels)):
    s_score = silhouette_score(features, labels[i])
    print(titles[i]+f' score: {s_score}')
# dbscan_clustering(features)
