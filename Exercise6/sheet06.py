import numpy as np
import numpy.random as random
from scipy.spatial.distance import cdist
from scipy import misc
from scipy.ndimage.morphology import binary_erosion
import matplotlib.pyplot as plt
from skimage import color, measure
import skimage.filters

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
        labels_c, n_components = measure.label(cluster_img_c, neighbors=n,
                return_num=True)
        labels += (cluster_img_c * count) + labels_c
        count += n_components
    return labels.astype(np.int) - 1

################################################################################
#                                    Task 1                                    #
################################################################################

# # this has been found to give a good result, but uses all 1000 iters
# random.seed(2)

# image = misc.imread('peppers.png')
# k = 5
# centers, labels = kmeans(image, k)
# label_image = label_image(labels, k, 4)

# plt.imshow(label_image, cmap='prism')
# plt.show()

# random.seed(0)
# image = color.rgb2hsv(misc.imread('peppers.png'))
# k = 6
# centers, labels = kmeans(image, k, channels=np.array([0,2]))
# for label in np.arange(k):
#     image[labels == label] = centers[label]

# plt.imshow(color.hsv2rgb(image))
# plt.show()

################################################################################
#                                    Task 2                                    #
################################################################################

def boundary(obj_image):
    return obj_image ^ binary_erosion(obj_image)

def region_saliency(image, labels):
    """Labels must be consecutive with no unused ones in between"""
    regions = np.unique(labels)
    C_k = np.zeros((regions.size, 3))
    for R_i in regions:
        C_k[R_i, :] = np.average(image[labels == R_i,:], axis=0)
    S_R = np.zeros(regions.size)
    for R_i in regions:
        B = boundary(labels == R_i)
        boundary_idx = np.argwhere(B)
        for x,y in boundary_idx:
            N_4 = [(i,j) for i,j in [(x+1,y), (x,y+1), (x-1,y), (x,y-1)]
                    if i >= 0 and i < image.shape[0] 
                    and j >= 0 and j < image.shape[1]]
            neighbor_regions = [labels[idx] for idx in N_4]
            N_diff = (neighbor_regions != R_i).sum()
            s2 = sum(np.linalg.norm(C_k[R_j,:]-C_k[R_i,:]) for R_j in
                    neighbor_regions)
            if N_diff > 1:
                S_R[R_i] += 1/N_diff * s2
        S_R[R_i] /= 1/B.size
    return S_R

# saliency = region_saliency(image, label_image)
# print(np.unique(label_image))
# print(saliency)


################################################################################
#                                    Task 3                                    #
################################################################################
from skimage.transform import hough_line
import matplotlib.pyplot as plt
import numpy as np
import operator

steps = lambda p,q : max(map(lambda x,y: abs(x-y), p, q))+1
coords = lambda p,q,s : [np.linspace(x,y,s,dtype=np.uint16) for x,y in zip(p,q)]

def point(img, p):
    "Insert a point in the black/white image at position p"
    img[p] = 1

def line(img, p, q):
    "Insert a line from p to q in the black/white image"
    img[coords(p,q,steps(p,q))] = 1

def polygon(img, vertices):
    "Insert a (closed) polygon given by a list of points into the black/white image"
    for p, q in zip(vertices, vertices[1:]+vertices[0:1]):
        line(img,p,q)


def my_hough_line(img, angles=180):
    """
    Apply linear Hough transform to the given image.
    """
    d_max = int(np.ceil(np.sqrt(sum(map(np.multiply,img.shape,img.shape)))))
    accumulator = np.zeros((2*d_max, angles), np.uint64)
    thetas = np.linspace(0, angles-1, angles, dtype=np.int)
    thetas_rad = np.deg2rad(thetas)
    cos_theta = np.cos(thetas_rad)
    sin_theta = np.sin(thetas_rad)

    edge_pts = np.argwhere(img)
    for x, y in edge_pts:
        d = x * cos_theta + y * sin_theta
        d_idx = np.round(d).astype(np.int) + d_max // 2
        accumulator[d_idx, thetas] += 1

    return accumulator


# img = np.zeros((100,100))


# # You may try different paintings here:
# # point(img, (10,10))
# line(img,(00,00),(99,99))
# line(img,(00,99),(99,00))
# # polygon(img,[(10,30),(50,50),(10,70)])



# plt.figure(figsize=(12, 4))
# plt.gray()
# plt.subplot(1,3,1) ; plt.title('Image'); plt.imshow(img)

# out, angles, d = hough_line(img)
# plt.subplot(1,3,2) ; plt.title('Hough transform (skimage)');
# plt.xlabel('Angles (degrees)')
# plt.ylabel('Distance (pixels)')
# plt.imshow(np.log(1 + out), extent=[np.rad2deg(angles[-1]), np.rad2deg(angles[0]), d[-1], d[0]])

# my_out = my_hough_line(img)
# plt.subplot(1,3,3) ; plt.title('Hough transform (own implementation)');
# plt.imshow(np.log(1+my_out))
# plt.show()

from skimage.transform import hough_line
import matplotlib.pyplot as plt
import numpy as np
import operator

        
def my_inverse_hough_line(accumulator, shape):
    """
    Compute an inverse Hough transform, i.e. compute the image from the accumulator space.
    """
    img = np.zeros(shape, np.uint64)

    return img

# img = np.zeros((100,100))

# #point(img, (10,10))
# #line(img,(10,20),(70,20))
# polygon(img,[(10,30),(50,50),(10,70)])


# plt.figure(figsize=(12, 4))
# plt.gray()
# plt.subplot(1,3,1) ; plt.title('Image'); plt.imshow(img)

# out, angles, d = hough_line(img)
# plt.subplot(1,3,2) ; plt.title('Hough transform (skimage)');
# plt.xlabel('Angles (degrees)')
# plt.ylabel('Distance (pixels)')
# plt.imshow(np.log(1 + out), extent=[np.rad2deg(angles[-1]), np.rad2deg(angles[0]), d[-1], d[0]])

# img2 = my_inverse_hough_line(out, img.shape)
# plt.subplot(1,3,3) ; plt.title('Inverse Hough transform');
# plt.imshow(img2)
# plt.show()

def sector_mask(shape,centre,radius,angle_range):
    """
    From http://stackoverflow.com/a/18354475/2397253
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 == radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def my_hough_circle(img, radius=10):
    """
    Apply linear Hough transform to the given image.
    """
    accumulator = np.zeros(img.shape, dtype=np.int)

    edge_pts = np.argwhere(img)
    for x, y in edge_pts:
        if (img.shape[0] - radius > x >= radius and img.shape[1] - radius > y >=
                radius):
            mask = sector_mask(img.shape, (x, y), radius, (0,360))
            accumulator[mask] += 1

    return accumulator

img = misc.imread("xmas.png")
hough = my_hough_circle(img, radius=30)
plt.gray()
plt.imshow(hough > 0.5 * hough.max())
plt.show()
