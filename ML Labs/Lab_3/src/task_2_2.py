import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets
from fileProcessor import *

np.random.seed(5)

file_name = "clustering_3"
X = return_features_labels("data/" + file_name + ".csv")

fig, sub = plt.subplots(2, 2)
ax = sub.flatten()
counter = 0
for min_samples in (2, 5, 10, 20):
    db = DBSCAN(eps=0.3, min_samples=min_samples)
    y_db = db.fit_predict(X)
    n_clusters_ = len(set(y_db)) - (1 if -1 in y_db else 0)

    for count in range(n_clusters_):
        ax[counter].scatter(X[y_db == count, 0], X[y_db == count, 1], s=25, label="cluster " + str(count))
    ax[counter].legend(scatterpoints=1)
    ax[counter].grid()
    ax[counter].set_title("Min samples = " + str(min_samples))
    counter += 1
    fig.suptitle("[ " + file_name + " ] Method = DBSCAN")

plt.show()
