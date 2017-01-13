# coding: utf-8
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

plt.gray()

img = misc.imread('dolly.png', mode = 'F')

img_ft = fft2(img)
img_ft_shift = fftshift(img_ft)

amplitude = np.abs(img_ft_shift)
phase = np.angle(img_ft)

# f, ax = plt.subplots(2,2)
# ax[0,0].set_title("Log amplitude information")
# ax[0,0].imshow(np.log(amplitude))
# ax[0,1].set_title("Log amplitude histogram")
# # histograms?? how do I get the power spectrum?
# ax[0,1].hist(amplitude.flatten(), bins=200, log=True)

# ax[1,0].set_title("Phase information")
# ax[1,0].imshow(phase)
# ax[1,1].set_title("Phase histogram")
# ax[1,1].hist(phase.flatten(), bins=200)

# f.tight_layout()
# f.show()


# **c)** Transform the image back from the frequency space to the image space (again using `fft2`).
# What do you observe? Explain and repair the result.

# There is nothing to observe or repair, the two look identical.
# plt.imshow(np.real(ifft2(np.abs(img_ft) * np.exp(1j * np.angle(img_ft)))))
# plt.show()

# f, axes = plt.subplots(1,2)
# f.tight_layout()
# axes[0].imshow(img)
# axes[0].set_title("Original")
# restored = ifft2(fft2(img))
# axes[1].imshow(np.real(restored))
# axes[1].set_title("ifft(fft(.))")
# plt.show()

# **d)** Now restore the image, but only keep the amplitude and vary the phase. Try fixed phase
# values (0, $\pi/2$,. . . ), a random phase matrix, or a noisy version of the original phase values.

def phase_amp_to_img(amplitude, phase):
    return np.real(ifft2(amplitude * np.exp(1j * phase)))

amplitude = np.abs(img_ft)
# phase_random = np.random.uniform(phase.min(), phase.max(), phase.shape)
# phase_disturbed = phase + np.random.normal(size=phase.shape)
# phase_constants = [np.full(phase.shape, value) for value in (k * np.pi / 2 for k
#     in range(0, 5))]

# f, ax = plt.subplots(2,4)
# f.delaxes(ax[-1,-1])
# f.tight_layout()
# images = [ phase_amp_to_img(amplitude, phase_random),
#         phase_amp_to_img(amplitude, phase_disturbed) ] + list(map(lambda p:
#             phase_amp_to_img(amplitude, p), phase_constants))
# titles = ['Random', 'Disturbed'] + ['Const %d*pi/2' % k for k in range(0, 5)]
# for i in range(0, 7):
#     ax[i//4, i % 4].imshow(images[i])
#     ax[i//4, i % 4].set_title(titles[i])
# plt.show()

# **e)** Do the same, but now keep the phase while varying the amplitude values. Again try constant,
# amplitude, randomly distributed amplitudes and noisy versions of the the original values.

def normalise(img):
    img = img - img.min()
    img = 255 * img / img.max()
    return img

max_amp = amplitude.max()
min_amp = amplitude.min()
step_size = np.floor((max_amp-min_amp) / 3)
amp_random = np.random.uniform(min_amp, max_amp, size=amplitude.shape)
amp_disturbed = amplitude + np.random.normal(0, amplitude.std()/4, size=amplitude.shape)
amp_constants = [np.full(amplitude.shape, v) for v in np.arange(min_amp, max_amp, step_size)]

f, ax = plt.subplots(2,3)
f.delaxes(ax[-1,-1])
f.tight_layout()
images = [ phase_amp_to_img(amp_random, phase),
        phase_amp_to_img(amp_disturbed, phase) ] + list(map(lambda a:
            phase_amp_to_img(a, phase), amp_constants))
titles = ['Random', 'Disturbed'] + ['Const %2.2f' % v for v in np.arange(min_amp, max_amp, step_size)]
for i in range(0, 5):
    ax[i//3, i % 3].imshow(normalise(images[i]))
    ax[i//3, i % 3].set_title(titles[i])
plt.show()




# *Hint:* Python and numpy can deal with complex numbers: `np.real()` and `np.imag()` provide the real and imaginary parts. `np.abs()` and `np.angle()` provide amplitude and phase. `np.conj()` gives the complex conjugate.

# # ## Exercise 2 (Implementing Fourier Transform – 8p)

# # **a)** 
# # Explain in your own words the idea of Fourier transform. What is the frequency space? What does a point in that space represent?
# FIXME: change cell type to 'Markdown' and put your answer here!
# # **b)** First implement a one-dimensional discrete version of Fourier transform, i.e. use the formula
# # $$ c_n = \sum_{x=0}^{L-1} f(x)\cdot e^{\tfrac{2\pi i\cdot n}{L}\cdot x} \qquad \text{for $n=0,\ldots,L-1$}$$
# # for complex valued coefficients.
# # 
# # Plot the graph and the results of your Fourier transform, using the Matplotlib function `plot()`, for different functions. Compare your results with the output of the function `numpy.fft.fft`.

# # In[ ]:

# import numpy as np
# from scipy import misc
# import matplotlib.pyplot as plt

# def fourier1d(func):
#     """
#     Perform a discrete 1D Fourier transform.
    
#     """
#     ft = np.zeros(func.shape, dtype = np.complex)

#     # FIXME: put your code here!


#     return ft

# # number of points
# L = np.arange(100)

# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# func = np.sin(2*np.pi*L/len(L))
# #func = np.zeros(L.shape); func[40:60] = 1;
# #func = gaussian(L, 0, 10)

# ft = fourier1d(func)
# #ft = np.fft.fft(func)
# plt.figure(figsize=(12,8))

# plt.gray()
# plt.subplot(1,3,1); plt.plot(L,func); plt.title('Function')
# plt.subplot(1,3,2); plt.plot(L,np.abs(ft)); plt.title('FT (Amplitude)')
# plt.subplot(1,3,3); plt.plot(L,np.angle(ft)); plt.title('FT (Frequency)')
# plt.show()


# # **c)** Now implement a 2-dimensional version of Fourier transform for images, using the formula from the lecture. Compare your result with the output of `fft2`.

# # In[ ]:

# import numpy as np
# from scipy import misc
# import matplotlib.pyplot as plt

# def fourier2d(img):
#     """
#     Perform discrete 2D Fourier transform of a given image.
#     """

#     ft = zeros(img.shape)

#     # FIXME: put your code here!


#     return ft

# img = misc.imread('dolly.png', mode = 'F')

# # FIXME: put your code here!


# plt.show()


# # ## Exercise 3 (Applying Fourier Transform – 6p)
# # 
# # 1. Read the image `text_deg.jpg`, display it and apply Fourier transform. The resulting amplitude should show the angle of the text.
# # 
# # 2. Try to automatically get the rotation angle from the Fourier space. There are different ways to achieve this.
# #    Hints:
# #    * You may apply an erosion operation to strengthen the text sections and thereby get
# #      better (i.e. less noisy) amplitude values.
# #    * You may threshold the amplitudes, to only keep “relevant” values. You can then compute the angle of the largest relevant value.
# #    * Alternatively, you may apply methods you know from other lectures to get the main component and compute its angle.
# # 
# # 3. Rotate the image back to its originally intended orientation (`scipy.misc.imrotate`).

# # In[ ]:

# import numpy as np
# from scipy import misc
# import matplotlib.pyplot as plt

# # FIXME: put your code here!

