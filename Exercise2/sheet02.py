# coding: utf-8
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import math

def global_contrast(img, range = None):
    """
    Compute the global contrast for a given image.
    """

    # determine range: simplified, just checks for uint8 
    if range is None:
        range = (0,255) if img.dtype == np.uint8 else (0.,1.)

    c = (np.max(img) - np.min(img)) / range[1]

    return c

def local_contrast(img):
    """
    Compute the local contrast for a given image. Here we just ignore the pixels
    that don't have a 4-neighbourhood.
    """

    contrast = 0
    for x in range(1,img.shape[0]-1): # what to do with the edges?
        for y in range(1,img.shape[1]-1):
            contrast += np.abs(img[x,y] - .25 * np.average(np.array([img[x-1,y], img[x+1,y], img[x,y-1], img[x,y+1]])))

    return contrast / ((img.shape[0] - 2) * (img.shape[1] - 2))
        

def entropy(img):

    n = img.size
    h, _ = np.histogram(img, 256) 
    h = h.astype('float') / n
    e = 0
    for freq in h:
        if freq > 1e-4: # == 0 for float values
            e -= freq * np.log2(freq) 
    return e

# img = imread('dark.png')
# plt.title("Image Entropy value: {}".format(entropy(img)))
# plt.imshow(img, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# plt.hist(img.flatten(),256,(0,255))
# plt.show()

def image_info(img):
    info = "global contrast: {}, local contrast: {}, entropy: {}"
    plt.title(info.format(global_contrast(img),local_contrast(img),entropy(img)))
    plt.imshow(img, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
    plt.show()


# # high global, low local
# img1 = np.zeros((256,256),np.uint8)
# img1[10,10] = 255

# # image_info(img1)

# # high local low global
# img2 = np.zeros((256,256),np.uint8)
# img2[::2,1::2] = 3 # set every other element so some nonzero value in both directions

# # image_info(img2)

# # I have no better ideas
# img3 = np.zeros((256,256),np.uint8)
# for column in range(256):
#     img3[column,:] = column
# img3 = (img3 / 10).astype(np.uint8)
# image_info(img3)

def he(img):
    """
    Apply histogram equalization (HE) to the image.
    
    img: numpy.ndarray (dtype=uint8)
        The image to be equalized.
        
    Returns
    -------
    equalized: numpy.ndarray (dtype=uint8)
        The equalized image.
    """
    
    equalized = np.zeros(img.shape, dtype=np.uint8)
    hist, _= np.histogram(img, 256)
    h = hist / img.size
    cumsum = h.cumsum()
    # for all gray values, change image in all pixels with that value
    for v in range(256):
        equalized[img == v] = np.ceil(256 * cumsum[v]) - 1
    return equalized


#img = imread('canada.png', mode = 'L')
# img = imread('dark.png', mode = 'L')

# plt.title("Image Entropy value: {}".format(entropy(img)))
# plt.imshow(img, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# plt.hist(img.flatten(),256,(0,255))
# plt.show()

# img2 = he(img)
# plt.title("Image Entropy value: {}".format(entropy(img2)))
# plt.imshow(img2, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# plt.hist(img2.flatten(),256,(0,255))
# plt.show()
# print(img2.min(),img2.max())


def ahe(img, size = (16,16)):
    """
    Apply adaptive histogram equalization (AHE) to the image.
    
    img: numpy.ndarray (dtype = uint8)
        The image to equalize.
    size: tuple of ints
        The size of the surrounding for which the histogram of each pixel is
        to be computed.
        
    Returns
    -------
    equalized: numpy.ndarray (dtype = uint8)
        The equalized image.
    """

    h         = size[0] // 2
    w         = size[1] // 2
    n         = img.shape[0]
    m         = img.shape[1]
    equalized = np.zeros(img.shape, np.uint8)
    for i in range(n):
        for j in range(m):
            region         = img[max(0, i-h):min(i+h, n), max(0, j-w):min(j+w, m)]
            equalized[i,j] = np.sum(region < img[i,j]) * 255 / region.size

    return equalized

img = imread('canada.png', mode='L')

# plt.title("Image Entropy value: {}".format(entropy(img)))
# plt.imshow(img, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# plt.hist(img.flatten(),256,(0,255))
# plt.show()

img2 = ahe(img, size=(8,8))
# plt.title("Image Entropy value: {}".format(entropy(img2)))
# plt.imshow(img2, cmap = plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# plt.hist(img2.flatten(),256,(0,255))
# plt.show()
