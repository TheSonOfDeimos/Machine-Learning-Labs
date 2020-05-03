import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
from fileProcessor import *

np.random.seed(5)
file_name = "clustering_2"
X = return_features_labels("data/" + file_name + ".csv")

fig, sub = plt.subplots(2, 2)
ax = sub.flatten()
counter = 0
for cluster_count in (2, 3, 4, 5):
    hk = AgglomerativeClustering(n_clusters=cluster_count)
    y_hk = hk.fit_predict(X)

    for count in range(cluster_count):
        ax[counter].scatter(X[y_hk == count, 0], X[y_hk == count, 1], s=25, label="cluster " + str(count))

    ax[counter].legend(scatterpoints=1)
    ax[counter].grid()
    ax[counter].set_title("Cluster count = " + str(cluster_count))
    counter += 1
    fig.suptitle("[ " + file_name + " ] Method = Hierarchical clustering")

plt.show()
