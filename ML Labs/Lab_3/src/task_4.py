from pandas import read_csv
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


def plot_dendrogram(model, **kwargs):
 children = model.children_
 distance = np.arange(children.shape[0])
 no_of_observations = np.arange(2, children.shape[0] + 2)
 linkage_matrix = np.column_stack(
     [children, distance, no_of_observations]).astype(float)
 dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
 data = read_csv("data/votes.csv", delimiter=",")
 data = data.fillna(data.mean()).values
 print([x.mean() for x in data][31])
 model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
 model.fit(data)
 plt.title('Hierarchical Clustering Dendrogram for votes.csv')
 plot_dendrogram(model)
 plt.show()
