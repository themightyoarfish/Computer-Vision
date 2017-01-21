################################################################################
#                                  Exercise 1                                  #
################################################################################
# import matplotlib
# from scipy.misc import imread
# from scipy.signal import convolve, fftconvolve
# from numpy.fft import fft2, ifft2
# from matplotlib import pyplot as plt
# import numpy as np

# def draw_box(img, center, halfwidth=30, val=0):
#     """Draw a box around center"""
#     m,n = img.shape
#     mx, my = center
#     l = halfwidth
#     img[max(0,mx-l):min(m,mx-l+2),max(0,my-l):min(n,my+l)] = val
#     img[max(0,mx-l):min(m,mx+l),max(0,my-l):min(n,my-l+2)] = val
#     img[max(0,mx+l-2):min(m,mx+l),max(0,my-l):min(n,my+l)] = val
#     img[max(0,mx-l):min(m,mx+l),max(0,my+l-2):min(n,my+l)] = val

# # load image
# img1 = imread('waldo/wheresWaldo1.jpg', mode='L')[600:1100,1400:1700]

# # zero-mean templates
# templates = [imread('waldo/waldo{}.jpg'.format(d), mode='L') for d in range(1,2)]
# templates = [t - t.mean() for t in templates]

# # array for correlation coefficients
# X, Y = img1.shape
# corr_coeffs = np.zeros((X, Y, len(templates)))

# # do fourier convolution
# conv = fftconvolve(img1, templates[0], mode='same')
# # the range of values is not [-1,1] ?????
# corr_coeffs[:,:,0] = conv / (img1.std() * templates[0].std())

# max_loc = np.unravel_index(np.argmax(corr_coeffs[:,:,0]), img1.shape)


# draw_box(img1, max_loc, 30)

# plt.figure()
# plt.imshow(img1, cmap='gray')
# plt.show()

################################################################################
#                                  Exercise 2                                  #
################################################################################
import numpy as np

def pca(data):
    """
    Perform principal component analysis.
    
    Arguments:
        data - an k*n dimensional array (k entries with n dimensions)
        
    Results:
        pc - an array holding the principal components as columns
    """
    # seems hard to believe, but this is actually sum(np.outer(row,row) for row in data)
    autocorr = np.dot(data.T, data)
    autocorr = autocorr / autocorr.mean()
    evals, evecs = np.linalg.eig(autocorr)
    
    return pc

################################################################################
#                                  Exercise 3                                  #
################################################################################
import sys
import os
import glob
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def read_images_from_directory(directory,suffix,shape):
    """
    Read all images found in DIRECTORY with given file
    name SUFFIX. All images should have the same SHAPE,
    specified as (rows,columns).
    
    Returns:
        images: A numpy array of shape m*rows*columns (from shape)
        names: A list of corresponding image names.
    """
    
    # initialize the image array and name list
    images = np.empty((0, *shape))
    names = []

    # now loop through all image files in the directory
    for file_name in glob.glob(directory + '/*.' + suffix):
        if os.path.isfile(file_name):

            # load each image (as double)
            img = misc.imread(file_name, mode = 'F')
            
            # check for correct size
            if img.shape == shape:
                images = np.append(images,img.reshape((1,*shape)),axis=0)
                names.append(os.path.basename(file_name))
            else:
                print('warning: Image "' + file_name + '" with wrong size will be ignored!', file=sys.stderr)
        
    return images, names


img_shape = (192 ,168);
train_imgs, train_names = read_images_from_directory('trainimg', 'pgm', img_shape)
m, n, k = train_imgs.shape
train_imgs_flat = np.reshape(train_imgs, (m, n * k))

evecs = pca(train_imgs_flat)
