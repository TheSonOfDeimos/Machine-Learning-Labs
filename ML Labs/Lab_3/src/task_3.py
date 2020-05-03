import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.cluster import AgglomerativeClustering
from time import time


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

n_colors = 2

image_name = "china"
image = load_sample_image(image_name + ".jpg")
image = np.array(image, dtype=np.float64) / 255
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))
image_array_sample = shuffle(image_array, random_state=0)[:1000]

fig, sub = plt.subplots(2, 3)
ax = sub.flatten()
counter = 0

ax[counter].set_title('Original image (96,615 colors)')
ax[counter].imshow(image)
ax[counter].axis('off')
counter += 1

for colors in (64, 32, 16, 8, 4):
    model = KMeans(n_clusters=colors, random_state=0).fit(image_array_sample)
    labels = model.predict(image_array)
    ax[counter].set_title("Quantized image (" + str(colors) + " colors, K-Means)")
    ax[counter].imshow(recreate_image(model.cluster_centers_, labels, w, h))
    ax[counter].axis('off')
    counter += 1





plt.show()
