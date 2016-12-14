import numpy as np
import numpy.random as random
from scipy.spatial.distance import cdist
from scipy import misc
import matplotlib.pyplot as plt

def kmeans(img, k):
    h, w, c = img.shape
    img_flat = np.reshape(img, (h*w, 3))
    epsilon = 0.001
    previous = random.uniform(low=img.min(), high=img.max(), size=(k,3))
    vectors = previous.copy()
    iter = 1
    labels = np.empty(img.shape)

    while True:
        pairwise_dists = cdist(img_flat, previous)
        min_indices = np.argmin(pairwise_dists, axis=1)
        labels = np.reshape(min_indices, (h, w))

        previous = vectors.copy()
        for kluster in np.arange(k):
            vectors[kluster,:] = img[labels == kluster].mean(axis=0)
            if not np.any(labels == kluster):
                vectors[kluster,:] = random.uniform(img.min(), img.max(), 3)
        if (not np.any(np.abs(vectors - previous) > epsilon)) or iter > 1000:
            break
        else:
            if iter % 10 == 0:
                print(iter)
            iter += 1

    print("Iterations %d" % iter)
    return vectors, labels

image = misc.imread('peppers.png')
k = 6
vectors, labels = kmeans(image, k)
for label in np.arange(k):
    image[labels == label] = vectors[label]

plt.imshow(image)
plt.show()
