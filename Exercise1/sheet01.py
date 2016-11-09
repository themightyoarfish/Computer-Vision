from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

image = misc.imread('./images/starship.png') / 255.

# a)
box_filter = np.ones((5,5))
image_box  = convolve(image, box_filter, mode = 'constant', cval = 0)

# b) we could manually create a filter, but we can also use scipy
gauss_filter      = np.zeros((5,5))
gauss_filter[2,2] = 1 # use one 1-Pixel to convolve
gauss_filter      = gaussian_filter(gauss_filter, sigma=2)
image_gauss       = convolve(image, gauss_filter, mode='constant', cval=0)

# c) sobel
sobel_filter = np.array([ [0, -1, -2], [1, 0, -1], [2, 1, 0] ]) * 0.25
image_sobel  = convolve(image, sobel_filter, mode                      = 'constant', cval = 0)

# d) laplace
laplace_filter = np.array([ [0, 1, 0], [1, -4, 1], [0, 1, 0] ])
image_laplace  = convolve(image, laplace_filter, mode           = 'constant', cval = 0)

f, axarr = plt.subplots(2,2)
axarr    = axarr.reshape(4)
f.tight_layout()
f.subplots_adjust(hspace=.05, wspace=.05)
axarr[0].imshow(image_box, cmap='gray')
axarr[1].imshow(image_gauss, cmap='gray')
axarr[2].imshow(image_sobel, cmap='gray')
axarr[3].imshow(image_laplace, cmap='gray')
plt.show()
