import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from fileProcessor import *

np.random.seed(5)
file_name = "clustering_3"
X = return_features_labels("data/" + file_name + ".csv")

fig, sub = plt.subplots(2, 2)
ax = sub.flatten()
counter = 0
for cluster_count in (2, 3, 4, 5):
    km = KMeans(n_clusters=cluster_count)
    y_km = km.fit_predict(X)

    for count in range(cluster_count):
        ax[counter].scatter(X[y_km == count, 0], X[y_km == count, 1], s=25, label="cluster " + str(count))

    ax[counter].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, marker='v', c='red', label='centroids')
    ax[counter].legend(scatterpoints=1)
    ax[counter].grid()
    ax[counter].set_title("Cluster count = " + str(cluster_count))
    counter += 1
    fig.suptitle("[ " + file_name + " ] Method = KMeans")

plt.show()
