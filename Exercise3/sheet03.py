import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph


def my_boundary(img):
    """
    Compute boundary of binary image.

    Parameters
    ----------
    img : ndarray of bools
        A binary image.
        
    Returns
    -------
    boundary : ndarray of bools
        The boundary as a binary image.
    """

    structuring_element = np.ones((3, 3))
    structuring_element[0, 0] = 0
    structuring_element[2, 2] = 0
    structuring_element[0, 2] = 0
    structuring_element[2, 0] = 0
    return img ^ morph.binary_erosion(img, structuring_element)


# img = plt.imread("engelstrompete.png") > 0
# plt.gray()
# plt.imshow(my_boundary(img))
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology as morph, generate_binary_structure


def my_distance_transform(img):
    """
    Distance transform of binary image.

    Parameters
    ----------
    img : ndarray of bools
        A binary image.
        
    Returns
    -------
    dt : ndarray of ints
        The distance transform of the input image.
    """

    dt1 = np.zeros(img.shape, np.int32)
    c = 0
    img_wo_boundary = img.copy()
    # while not completely eroded
    while np.any(img_wo_boundary):
        # find boundary
        boundary = my_boundary(img_wo_boundary)
        img_wo_boundary ^= boundary  # shave of current boundary
        dt1[boundary] = c  # boundary can be used as mask
        c += 1  # make brighter with every step

    return dt1


def my_generalized_distance_transform(img, **kwargs):
    dt1 = np.zeros(img.shape, np.int32)
    c = 1
    img_wo_boundary = img.copy()
    # while not completely eroded
    while np.any(img_wo_boundary):
        # find boundary
        boundary = my_boundary(img_wo_boundary, **kwargs)
        img_wo_boundary ^= boundary  # shave of current boundary
        dt1[boundary] = c  # boundary can be used as mask
        c += 1  # make brighter with every step

    dt2 = np.zeros(img.shape, np.int32)
    c = 1
    img_wo_boundary = ~img.copy()
    # while not completely eroded
    while np.any(img_wo_boundary):
        # find boundary
        boundary = my_boundary(img_wo_boundary, **kwargs)
        img_wo_boundary ^= boundary  # shave of current boundary
        dt2[boundary] = c  # boundary can be used as mask
        c += 1  # make brighter with every step

    result = dt1 - dt2
    # m = result.min()
    # result = result + (- m if m < 0 else m)
    return result / result.max()


# img = plt.imread("engelstrompete.png") > 0
# dt = my_distance_transform(img)
# plt.gray()
# plt.imshow(dt+50*img)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph


def my_morph(A, B, ratio):
    """
    Morphing from binary image A to image B.

    Parameters
    ----------
    A : ndarray of bools
        A binary image (start).
    B : ndarray of bools
        A binary image (target), same shape as A.
    ratio : float from 0.0 to 1.0
        The ratio of image A and image B.
        0.0=only image A, 1.0=only image B.
        
    Returns
    -------
    morph : ndarray of bools
        A binary intermediate image between A and B.
    """

    d_a = my_distance_transform(A)
    d_b = my_distance_transform(B)

    result = (ratio * d_b + (1 - ratio) * d_a)
    return result > 0


# img1 = plt.imread("kreis.png") > 0
# img2 = plt.imread("engelstrompete.png") > 0

# plt.gray()
# for i, ratio in enumerate(np.linspace(0, 1, 6), 1):
#     plt.subplot(2, 3, i)
#     plt.imshow(my_morph(img1, img2, ratio))
#     plt.axis('off')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph


def my_skeletonize(img):
    """
    Compute the skeloton of a binary image using hit_or_miss operator.
    
    Parameters
    ----------
    img : ndarray of bools
        Binary image to be skeletonized.
    
    Returns
    -------
    skeleton : ndarray of bools
        The skeleton of the input image.
    """

    # no idea why this works  ̄\_(ツ)_/ ̄
    # these are the elements from the slides. Don't care are 0s, meaning
    # whatever the image at such a location is, it does not matter. We need to
    # form the complementary kernels by hand since the Don't cares would
    # otherwise be set to 1
    element1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]])
    element1_c = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    element2 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    element2_c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    elements = [
        element1,
        np.fliplr(element1), element1.T,
        np.flipud(element1.T), element2,
        np.fliplr(element2.T),
        np.fliplr(element2), element2.T
    ]

    elements_c = [
        element1_c,
        np.fliplr(element1_c), element1_c.T,
        np.flipud(element1_c.T), element2_c,
        np.fliplr(element2_c.T),
        np.fliplr(element2_c), element2_c.T
    ]

    # this iteratively removes the pieces of the image for which the structuring
    # element hits (why does it work #clueless)
    skeleton = img.copy()
    while True:
        last = skeleton
        for (s1, s2) in zip(elements, elements_c):
            # this computes parts we can erase
            hm = morph.binary_hit_or_miss(skeleton, s1, s2)
            skeleton = skeleton & ~hm  # and not X removes X's elements from img
        if np.all(skeleton == last):  # end if nothing more to erase
            break
    return skeleton


img = plt.imread("engelstrompete.png") > 0
skel = my_skeletonize(img)
result = morph.distance_transform_cdt(img, metric='taxicab') + 50 * img
result[morph.binary_dilation(skel)] = 0
plt.gray()
plt.imshow(result)
plt.show()
