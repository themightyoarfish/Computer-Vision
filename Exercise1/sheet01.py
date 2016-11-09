from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

## Exercise 3
image = misc.imread('./images/starship.png') / 255.

# a)
box_filter = 1/25 * np.ones((5,5))
image_box  = convolve(image, box_filter, mode='constant', cval=0)

# b) we could manually create a filter, but we can also use scipy
gauss_filter      = np.zeros((5,5))
gauss_filter[2,2] = 1 # use one 1-Pixel to convolve
gauss_filter      = gaussian_filter(gauss_filter, sigma=2)
image_gauss       = convolve(image, gauss_filter, mode='constant', cval=0)

# c) sobel
sobel_filter = np.array([ [0, -1, -2], [1, 0, -1], [2, 1, 0] ]) * 0.25
image_sobel  = convolve(image, sobel_filter, mode='constant', cval=0)

# d) laplace
laplace_filter = np.array([ [0, 1, 0], [1, -4, 1], [0, 1, 0] ])
image_laplace  = convolve(image, laplace_filter, mode='constant', cval=0)

f, axarr = plt.subplots(2,2)
axarr    = axarr.reshape(4)
f.tight_layout()
f.subplots_adjust(hspace=.05, wspace=.05)
axarr[0].imshow(image_box, cmap='gray')
axarr[1].imshow(image_gauss, cmap='gray')
axarr[2].imshow(image_sobel, cmap='gray')
axarr[3].imshow(image_laplace, cmap='gray')
plt.show()

## Exercise 3
def my_convolve2d(img, kern):
    """Convolve an image with a kernel.

    img -- the image, provided as a two-dimensional array
    kern -- the kernel, also a two-dimensional array
    """
    
    # store the image size for easier access
    M,N = img.shape
    # store the kernel size
    m,n = kern.shape
    # and also the half kernel size
    mh, nh = (m//2, n//2)
    
    # Compute the convolution
    # Could do this with for-loops as well. We simply ignore all pixels whose
    # neighbourhood does not fully overlap the kernel. Result image will thus
    # be smaller
    # What is happening? For each row_index k except the mh top and bottom ones,
    # for each col_index l except the nh left and right ones, select the pixel's
    # neighbourhood (k-mh to k+mh horizontal, l-nh to l+nh vertical) and compute
    # elementwise product with kernel, then it up. That's the new value for that
    # pixel. result is then a list where the rows of the image are appended one
    # after another. We can simply reshape the list.
    result = [np.sum(kern * image[k-mh:k+mh+1,l-nh:l+nh+1]) for k in range(mh,M-nh) for l in
            range(nh,N-nh)]
    result = np.reshape(result, (M-2*mh, N-2*nh))

    return result

# Apply your function to an image:
# Try different filters, compare the results with Assignment 2

# Load the image
image = misc.imread('./images/starship.png', 'F')

box_3           = 1/9 * np.asarray([[1,1,1],[1,1,1],[1,1,1]])
laplace_3       = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
filtered_image  = my_convolve2d(image, box_3)
filtered_image2 = my_convolve2d(image, laplace_3)

plt.figure()
plt.imshow(filtered_image2, cmap = plt.get_cmap('gray'))
plt.show()
