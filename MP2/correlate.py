import skimage.io as io
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from skimage.color import rgb2gray


# %% Correlation function and Gaussean kernel
# Assumes that the image is already vectorized.
def MyCCorrelation_201630945_201632096(image, kernel, boundary_condition):
    valid_conditions = ['fill', 'valid', 'symm']
    # Boundary condition must be valid.
    if boundary_condition not in valid_conditions:
        raise Exception('The boundary condition isn\'t valid (must be \'fill\', \'valid\' or \'symm\').')
    # Default padding fill value assigned as 0.
    fill = 0
    # Transforms the image and kernel into a Numpy array to avoid index issues.
    img = np.asarray(image)
    ker = np.asarray(kernel)
    # Imagen M x N, mask m x n.
    M, N = np.size(img, 0), np.size(img, 1)
    m, n = np.size(ker, 0), np.size(ker, 1)
    # Result is the same size as the image.
    CCorrelation = np.zeros([M, N])
    a, b = int((m - 1) / 2), int((n - 1) / 2)
    # To better understand these iterations, they are thought as if the matrix was on a cartesian map where
    # the y axis is pointing down instead of up, therefore, y and x axis are the rows and columns, respectively.
    # The indexes j and i are used to loop through what would be the overlapped mask over the image. They are
    # also used to loop through the mask's weights by rescaling the index with respect to the current (x,y) pixel.
    for y in range(M):
        for x in range(N):
            s = 0
            j = y - a
            while j < y + a + 1:
                i = x - b
                while i < x + b + 1:
                    # 'Valid' boundary condition will skip over any pixel
                    # if the mask's boundaries leave the image's boundary.
                    if boundary_condition == 'valid' and (j < 0 or j >= M or i < 0 or i >= N):
                        s = img[y, x]
                        j = y + a
                        break
                    # If the mask's boundaries leave the image's boundary,
                    # it multiplies the assigned fill by the mask's weight.
                    elif boundary_condition == 'fill' and (j < 0 or j >= M or i < 0 or i >= N):
                        s += fill * ker[j - y + a, i - x + b]

                    # Now, these following conditions covers both the symmetric padding case and the regular
                    # case where the mask's boundary doesn't leave the image's boundary.
                    elif j < 0:
                        if boundary_condition == 'symm':
                            if i <= 0:
                                s += img[0, 0] * ker[j - y + a, i - x + b]
                            elif i >= N - 1:
                                s += img[0, N - 1] * ker[j - y + a, i - x + b]
                            else:
                                s += img[0, i] * ker[j - y + a, i - x + b]
                    elif j >= M:
                        if boundary_condition == 'symm':
                            if i <= 0:
                                s += img[M - 1, 0] * ker[j - y + a, i - x + b]
                            elif i >= N - 1:
                                s += img[M - 1, N - 1] * ker[j - y + a, i - x + b]
                            else:
                                s += img[M - 1, i] * ker[j - y + a, i - x + b]
                    else:
                        if i < 0:
                            if boundary_condition == 'symm':
                                s += img[j, 0] * ker[j - y + a, i - x + b]
                        elif i >= N:
                            if boundary_condition == 'symm':
                                s += img[j, N - 1] * ker[j - y + a, i - x + b]
                        else:
                            # This is the regular clase mentioned above.
                            s += img[j, i] * ker[j - y + a, i - x + b]
                    i += 1
                j += 1
            CCorrelation[y, x] = s
    return CCorrelation


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


# %% Processing of the roses image.
# The respective kernels for the laboratory.
base_kernels = [
    np.ones([3, 3]),
    np.ones([3, 3]) / 9,
    np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    np.asarray([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
]
roses = io.imread('roses.jpg')
roses_gry = rgb2gray(roses)
# Second kernel, the average kernel.
roses_ker = base_kernels[3]
roses_res = MyCCorrelation_201630945_201632096(roses_gry, roses_ker, 'symm')
# The method returns the image with added boundary, therefore it must be removed.
roses_res_2 = sig.correlate2d(roses_gry, roses_ker, boundary='symm')[1:-1, 1:-1]
err = np.mean(np.power(roses_res - roses_res_2, 2)) * 100
print(f'El error cuadrático medio es de {err}%.')

# %% Visualization of the results.
fig, axes = plt.subplots(1, 2)
fig.suptitle('Imagen original y resultado de cross-correlación con el kernel 3d')
axes[0].set_title('Original', fontsize=10)
axes[0].imshow(roses_gry, cmap='gray')
axes[0].axis('off')
axes[1].set_title('Kernel 3d (symmetric boundary)', fontsize=10)
axes[1].imshow(roses_res, cmap='gray')
axes[1].axis('off')
fig.show()
fig.savefig('roses_sample.png')
# input('Press enter to continue...')

# %% Testing additional kernels.
fig, axes = plt.subplots(1, 3)
fig.suptitle('Imagen original y resultado de cross-correlación con dos kernels')
roses_kern_a = MyCCorrelation_201630945_201632096(roses_gry, base_kernels[0], 'fill')
roses_kern_b = MyCCorrelation_201630945_201632096(roses_gry, base_kernels[1], 'fill')
print(f'Nivel de gris promedio de la imagen con kernel 3a: {np.sum(roses_kern_a) / np.size(roses_kern_a)}.')
print(f'Nivel de gris promedio de la imagen con kernel 3b: {np.sum(roses_kern_b) / np.size(roses_kern_b)}.')
axes[0].set_title('Original')
axes[0].imshow(roses_gry, cmap='gray')
axes[0].axis('off')
axes[1].set_title('Kernel 3a')
axes[1].imshow(roses_kern_a, cmap='gray')
axes[1].axis('off')
axes[2].set_title('Kernel 3b')
axes[2].imshow(roses_kern_b, cmap='gray')  # Already previously calculated.
axes[2].axis('off')
fig.show()
fig.savefig('roses_kernels.png')
# input('Press enter to continue...')

# %% Gaussian Kernel
fig, axes = plt.subplots(1, 2)
gauss_kernel = gaussian_kernel(5, 1)
roses_gauss = MyCCorrelation_201630945_201632096(roses_gry, gauss_kernel, 'fill')
fig.suptitle('Imagenes resultado de cross-correlación con kernel 3b y Gaussiano')
axes[0].set_title('Kernel 3b (promedio)')
axes[0].imshow(roses_kern_b, cmap='gray')
axes[0].axis('off')
axes[1].set_title('Kernel Gaussiano')
axes[1].imshow(roses_gauss, cmap='gray')
axes[1].axis('off')
fig.show()
fig.savefig('roses_3b_gauss.png')
# input('Press enter to continue...')

# %% Gaussian kernels with fixed size and variable sigma.
sigma_values = [1, 4, 10]
fig, axes = plt.subplots(1, 3)
fig.suptitle('Imagenes resultado de la cross-correlación con distintos kernels Gaussianos')
for ind, i in enumerate(sigma_values):
    gauss_kernel = gaussian_kernel(9, i)
    roses_gauss = MyCCorrelation_201630945_201632096(roses_gry, gauss_kernel, 'fill')
    axes[ind].set_title(r'Kernel Gaussiano $\sigma = $ ' + str(i), fontsize='small')
    axes[ind].imshow(roses_gauss, cmap='gray')
    axes[ind].axis('off')
fig.show()
fig.savefig('gaussian_roses_fixed_size.png')

# %% Gaussian kernels with fixed size and variable sigma.
size_values = [3, 5, 11]
fig, axes = plt.subplots(1, 3)
fig.suptitle('Imagenes resultado de la cross-correlación con distintos kernels Gaussianos')
for ind, i in enumerate(size_values):
    gauss_kernel = gaussian_kernel(i, 5)
    roses_gauss = MyCCorrelation_201630945_201632096(roses_gry, gauss_kernel, 'fill')
    axes[ind].set_title('Kernel Gaussiano ' + str(i) + r'$\times$' + str(i), fontsize='small')
    axes[ind].imshow(roses_gauss, cmap='gray')
    axes[ind].axis('off')
fig.show()
fig.savefig('gaussian_roses_fixed_sigma.png')

