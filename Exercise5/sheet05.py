import numpy as np
from scipy import misc
from scipy.ndimage.filters import convolve, convolve1d
import matplotlib.pyplot as plt

img = misc.imread('mermaid.png', mode='F')

pyramid_image = img
h, w = img.shape

filter1d_down = 1/16 * np.array([1,4,6,4,1])
kernel = np.outer(filter1d_down, filter1d_down)
plt.gray()

def reduce(img):
    current_level = convolve(img, kernel)
    return 255 * current_level[::2,::2] / current_level.max()
    
reduced_levels = []
current_level = img
while True:
    current_level = reduce(current_level)
    if current_level.size < 4:
        break
    reduced_levels.append(current_level.copy())
    pyramid_image[-current_level.shape[0]:,:current_level.shape[1]] = current_level


# plt.imshow(pyramid_image)
# plt.show()


def expand(img, new_size=None):
    # This is what opencv pyrUp does. No idea how to do it myself. 
    # Quote: "The function performs the upsampling step of the 
    # Gaussian pyramid construction, though it can actually 
    # be used to construct the Laplacian pyramid. First, it 
    # upsamples the source image by injecting even zero rows 
    # and columns and then convolves the result with the same 
    # kernel as in pyrDown() multiplied by 4."
    new_size = new_size or (2*img.shape[0], 2*img.shape[1])
    image_up = np.zeros(new_size) # odd number of rows or cols will get rounded
    # down during reduce. Here we simply make the output as big as desired,
    # hopefully no one will notice
    image_up[::2, ::2] = img
    return convolve(image_up, 4 * kernel, mode='constant')


img = misc.imread('mermaid.png', mode='F')
expanded_levels = []
for r in reduced_levels:
    current_level = expand(r)
    # plt.imshow(current_level)
    # plt.show()
    expanded_levels.append(current_level.copy())

pyramid_image = img
current_level = img
while True:
    sz = current_level.shape
    reduced = reduce(current_level)
    expanded = expand(reduced, new_size=sz)
    if current_level.size < 4:
        break
    pyramid_image[-current_level.shape[0]:,:current_level.shape[1]] = expanded - current_level
    current_level = reduced

# plt.imshow(pyramid_image)
# plt.show()

################################################################################
#                                  Exercise 2                                  #
################################################################################

img = misc.imread('mermaid.png', mode='L')

def get_patch(img, x, y, size=40):
    """
    Extract a rectangular patch from an image and mark it in the original image.
    """
    result = img[x:x+size,y:y+size].copy()
    img[x:x+size, [y,y+1,y+size,y+size+1]] = 0
    img[[x,x+1,x+size,x+size+1], y:y+size] = 0
    return result

patches = []
patches.append(get_patch(img, 50,130))
patches.append(get_patch(img, 110,80))
patches.append(get_patch(img, 260,340))
patches.append(get_patch(img, 310,110))
patches.append(get_patch(img, 100,440))


from itertools import product
def cooccurrence(img, dx=1, dy=1):
    """
    Compute a co-occurence matrix for the given image.
    
    Args:
        img          the grayscale image (uint8)
        dx,dy        the offset between the two reference points

    Returns:
        matrix       the co-occurence matrix
    """
    matrix = np.zeros((256,256))
    h, w = img.shape
    assert (dx, dy) != (0, 0), "Directional vector cannot be zero length."

    # first we computed index array to shave off the pixels which cannot have
    # neihbours in the given direction at the given distance
    if dx > 0 and dy > 0:
        indices = (np.arange(0, h-dx), np.arange(0, w-dy))
        indices_shift = (np.arange(dx, h), np.arange(dy, w))
    elif dx == 0 and dy > 0:
        indices = (np.arange(0, h), np.arange(0, w-dy))
        indices_shift = (np.arange(0, h), np.arange(dy, w))
    elif dx > 0 and dy == 0:
        indices = (np.arange(0, h-dx), np.arange(0, w))
        indices_shift = (np.arange(dx, h), np.arange(0, w))
    elif dx == 0 and dy < 0:
        indices = (np.arange(0, h), np.arange(-dy, w))
        indices_shift = (np.arange(0, h), np.arange(0, w-dy))
    elif dx < 0 and dy == 0:
        indices = (np.arange(-dx, h), np.arange(0, w))
        indices_shift = (np.arange(0, h-dx), np.arange(0, w))
    elif dx < 0 and dy < 0:
        indices = (np.arange(-dx, h), np.arange(-dy, w))
        indices_shift = (np.arange(0, h+dx), np.arange(0, w+dy))
    else:
        raise Exception("I can't deal with this situation.")

    def delta(arr):
        return arr == 0

    # we only need to loop over the values which actually occur in the patch
    for (g1, g2) in product(np.arange(img.min(), img.max()+1), np.arange(img.min(), img.max()+1)):

        # apparently, using integer arrays as indices is different from slices,
        # for some unknown reason. Dunno what it does, but np.ix_ yields arrays
        # that do what you would expect
        patch_g1 = delta(img[np.ix_(*indices)] - g1)
        patch_g2 = delta(img[np.ix_(*indices_shift)] - g2)

        # patch_g1.size == patch_g2.size
        matrix[g1, g2] = 1/patch_g1.size * (patch_g1 * patch_g2).sum()

    return matrix


# plt.figure(figsize=(12, 12))
plt.gray()
plt.imshow(img)
plt.show()


plt.figure(figsize=(12, 12))
i = 0
for p in patches:
    plt.subplot(len(patches),3,i+1); plt.axis('off'); plt.imshow(p)
    plt.subplot(len(patches),3,i+2); plt.imshow(cooccurrence(p,1,0))
    plt.subplot(len(patches),3,i+3); plt.imshow(cooccurrence(p,0,1))
    i += 3
plt.show()
