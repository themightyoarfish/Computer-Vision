import numpy as np
import numpy.random as random
from scipy.spatial.distance import cdist
from scipy import misc
import matplotlib.pyplot as plt
from skimage import color, measure

def batch_euclid(X,Y, channels):
    return np.sqrt(np.sum(np.square(X[:,channels] - Y[:,channels]), axis=1))

def kmeans(img, k, channels=np.array([0,1,2])):
    h, w, c = img.shape
    img_flat = np.reshape(img, (h*w, 3))
    epsilon = 1 # one rgb value minimal motion
    previous = random.uniform(low=img.min(), high=img.max(), size=(k,3))
    centers = previous.copy()
    iter = 1
    labels = np.empty(img.shape)

    while True:
        pairwise_dists = cdist(img_flat, previous)
        min_indices = np.argmin(pairwise_dists, axis=1)
        labels = np.reshape(min_indices, (h, w))

        previous = centers.copy()
        for kluster in np.arange(k):
            centers[kluster,:] = img[labels == kluster].mean(axis=0)
            if not np.any(labels == kluster):
                centers[kluster,:] = random.uniform(img.min(), img.max(), 3)
        distances = batch_euclid(centers, previous, channels)
        if (not np.any(distances > epsilon)) or iter > 999:
            break
        else:
            iter += 1

    print("Iterations: %d" % iter)
    return centers, labels

def label_image(cluster_img, k, n=4):
    labels = np.zeros(cluster_img.shape)
    count = 0
    for c in range(k):
        cluster_img_c = cluster_img == c
        labels_c = measure.label(cluster_img_c, neighbors=n)
        n_components = np.unique(labels_c).size
        labels += (cluster_img_c * count) + labels_c
        count += n_components
    return labels

################################################################################
#                                    Task 1                                    #
################################################################################

# this has been found to give a good result, but uses all 1000 iters
random.seed(2)

image = misc.imread('peppers.png')
k = 5
centers, labels = kmeans(image, k)
image = label_image(labels, k, 4)

plt.imshow(image, cmap='prism')
plt.show()

# random.seed(0)
# image = color.rgb2hsv(misc.imread('peppers.png'))
# k = 6
# centers, labels = kmeans(image, k, channels=np.array([0,2]))
# for label in np.arange(k):
#     image[labels == label] = centers[label]

# plt.imshow(color.hsv2rgb(image))
# plt.show()
