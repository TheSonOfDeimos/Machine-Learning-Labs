import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from fileProcessor import *

np.random.seed(5)

X = return_features_labels("data/pluton.csv")
cluster_count = 3

for n_init in (1, 5, 10):
    fig, sub = plt.subplots(2, 2)
    ax = sub.flatten()
    counter = 0

    for max_iter in (1, 100, 300, 1000):
        km = KMeans(n_clusters=cluster_count, max_iter=max_iter, n_init=n_init)
        y_km = km.fit_predict(X)

        for count in range(cluster_count) :
            ax[counter].scatter(X[y_km == count, 0], X[y_km == count, 1], s=25, label="cluster " + str(count))

        ax[counter].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, marker='v', c='red', label='centroids')
        ax[counter].legend(scatterpoints=1)
        ax[counter].grid()
        ax[counter].set_title("Max iter = " + str(max_iter))
        counter += 1
    fig.suptitle("N init = " + str(n_init))
    plt.show()

