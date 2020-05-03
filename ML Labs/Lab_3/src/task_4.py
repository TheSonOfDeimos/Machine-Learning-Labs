import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from fileProcessor import *


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = return_features_labels("data/votes.csv")

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.figure(1)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.figure(2)
y_hk = model.fit_predict(X)
n_clusters_ = len(set(y_hk)) - (1 if - 1 in y_hk else 0)
print("Cluster = " + str(n_clusters_))
for count in range(n_clusters_):
    plt.scatter(X[y_hk == count, 0], X[y_hk == count, 1], s=25, label="cluster " + str(count))

plt.legend(scatterpoints=1)
plt.grid()
plt.show()
