import skimage.io as io
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.color import rgb2gray


# %% Function MyAdaptMedian


def MyAdaptMedian_201630945_0(gray_image, window_size, max_window_size):
    # Transforms the image into a Numpy array to avoid index issues.
    img = np.asarray(gray_image)
    # Image M x N, same as the output.
    m, n = np.size(img, 0), np.size(img, 1)
    filtered_image = np.zeros([m, n])
    # To better understand these iterations, they are thought as if the matrix was on a cartesian map where
    # the y axis is pointing down instead of up, therefore, y and x axis are the rows and columns, respectively.
    for y in range(m):
        for x in range(n):
            # The 'processed' flag indicates when the value of pixel (j, i) has been determined.
            processed = False
            curr_wind_size = window_size
            a = int(curr_wind_size / 2)
            value = img[y, x]  # Placeholder; assigned during the cycle.
            while not processed and curr_wind_size <= max_window_size:
                # Since there's no boundary management, the window is extracted as is based on limits.
                # that is, there's no need to check for padding or symmetric values.
                window = img[max(0, y - a):min(y + a + 1, m), max(0, x - a):min(x + a + 1, n)]
                z_min, z_max, z_med = np.min(window), np.max(window), np.median(window)
                if (z_med - z_min) > 0 > (z_med - z_max):
                    value = img[y, x] if (img[y, x] - z_min) > 0 > (img[y, x] - z_max) else z_med
                    processed = True
                else:
                    curr_wind_size += 2
                    if curr_wind_size > max_window_size:
                        value = z_med
                        processed = True
            filtered_image[y, x] = value
    return filtered_image


# %% 6.1.2
# Returns a sample of size sample_size of the center of the image.
def get_sample(sample_size, image):
    s, m, n = int(sample_size / 2), np.size(image, 0), np.size(image, 1)
    y, x = int(m / 2), int(n / 2)
    return image[max(0, y - s):min(y + s + 1, m), max(0, x - s):min(x + s + 1, n)]


# Visual analysis of the noise present in the images.
noisy1 = io.imread('noisy1.jpg')
noisy2 = io.imread('noisy2.jpg')
noisy1_grey = rgb2gray(noisy1)
noisy2_grey = rgb2gray(noisy2)
sample_base = 31
sample1 = get_sample(sample_base, noisy1_grey)
sample2 = get_sample(sample_base, noisy2_grey)

# %% Plotting
fig, axes = plt.subplots(1, 2)
fig.suptitle(f'Región {sample_base}x{sample_base} central de las imágenes')
axes[0].set_title('Imagen noisy1')
axes[0].imshow(sample1, cmap='gray')
axes[0].axis('off')
axes[1].set_title('Imagen noisy2')
axes[1].imshow(sample2, cmap='gray')
axes[1].axis('off')
fig.show()
fig.savefig('noisy_noise_sample.png')
# input('Press enter to continue...')
fig, axes = plt.subplots(1, 2)
fig.suptitle(f'Imágenes en escala de grises')
axes[0].set_title('Imagen noisy1')
axes[0].imshow(noisy1_grey, cmap='gray')
axes[0].axis('off')
axes[1].set_title('Imagen noisy2')
axes[1].imshow(noisy2_grey, cmap='gray')
axes[1].axis('off')
fig.show()
fig.savefig('noisy_imagenes.png')
# input('Press enter to continue...')

# %%
window_sizes = [3, 5, 7]
maximum_window_size = 11
fig, axes = plt.subplots(2, len(window_sizes) + 1)
fig.suptitle('Resultado del filtro mediano adaptativo para la imagen noisy1')
axes[0, 0].set_title('Original')
axes[0, 0].imshow(noisy1_grey, cmap='gray')
axes[0, 0].axis('off')
axes[1, 0].imshow(sample1, cmap='gray')
axes[1, 0].axis('off')
for i, w in enumerate(window_sizes):
    axes[0, i + 1].set_title(f'{w}x{w}')
    result = MyAdaptMedian_201630945_0(noisy1_grey, w, maximum_window_size)
    axes[0, i + 1].imshow(result, cmap='gray')
    axes[0, i + 1].axis('off')
    axes[1, i + 1].imshow(get_sample(sample_base, result), cmap='gray')
    axes[1, i + 1].axis('off')
fig.show()
fig.savefig('noisy1_filter_median.png')
# input('Press enter to continue...')
fig, axes = plt.subplots(2, len(window_sizes) + 1)
fig.suptitle('Resultado del filtro mediano adaptativo para la imagen noisy2')
axes[0, 0].set_title('Original')
axes[0, 0].imshow(noisy2_grey, cmap='gray')
axes[0, 0].axis('off')
axes[1, 0].imshow(sample2, cmap='gray')
axes[1, 0].axis('off')
for i, w in enumerate(window_sizes):
    axes[0, i + 1].set_title(f'{w}x{w}')
    result = MyAdaptMedian_201630945_0(noisy2_grey, w, maximum_window_size)
    axes[0, i + 1].imshow(result, cmap='gray')
    axes[0, i + 1].axis('off')
    axes[1, i + 1].imshow(get_sample(sample_base, result), cmap='gray')
    axes[1, i + 1].axis('off')
fig.show()
fig.savefig('noisy2_filter_median.png')


# input('Press enter to continue...')

# %% Testing Gaussian filtering with the previously defined functions


# Testing will be done on the image noisy2 since it has gaussian noise.
# Full documentation can be found on the first delivery.
def my_cross_correlation(image, kernel, boundary_condition):
    valid_conditions = ['fill', 'valid', 'symm']
    if boundary_condition not in valid_conditions:
        raise Exception('The boundary condition isn\'t valid (must be \'fill\', \'valid\' or \'symm\').')
    fill = 0
    img = np.asarray(image)
    ker = np.asarray(kernel)
    M, N = np.size(img, 0), np.size(img, 1)
    m, n = np.size(ker, 0), np.size(ker, 1)
    c_correlation = np.zeros([M, N])
    a, b = int((m - 1) / 2), int((n - 1) / 2)
    for y in range(M):
        for x in range(N):
            s = 0
            j = y - a
            while j < y + a + 1:
                i = x - b
                while i < x + b + 1:
                    if boundary_condition == 'valid' and (j < 0 or j >= M or i < 0 or i >= N):
                        s = img[y, x]
                        j = y + a
                        break
                    elif boundary_condition == 'fill' and (j < 0 or j >= M or i < 0 or i >= N):
                        s += fill * ker[j - y + a, i - x + b]
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
                            s += img[j, i] * ker[j - y + a, i - x + b]
                    i += 1
                j += 1
            c_correlation[y, x] = s
    return c_correlation


def gaussian_kernel(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


test_sizes = [3, 5, 7, 9]
test_sigmas = [0.5, 1, 2, 4, 8]
fig1, axes1 = plt.subplots(len(test_sizes), len(test_sigmas))
fig2, axes2 = plt.subplots(len(test_sizes), len(test_sigmas))
fig1.suptitle('Filtered result of noisy1 with different gaussian kernels')
fig2.suptitle('Result samples of noisy1 with different gaussian kernels')
for ind_i, i in enumerate(test_sizes):
    for ind_j, j in enumerate(test_sigmas):
        gauss_kernel = gaussian_kernel(i, j)
        result = my_cross_correlation(noisy1_grey, gauss_kernel, 'fill')
        axes1[ind_i, ind_j].set_title(f'size: {i}, ' + r'$\sigma=$' + f'{j}', fontsize='small')
        axes1[ind_i, ind_j].imshow(result, cmap='gray')
        axes1[ind_i, ind_j].axis('off')
        axes2[ind_i, ind_j].set_title(f'size: {i}, ' + r'$\sigma=$' + f'{j}', fontsize='small')
        axes2[ind_i, ind_j].imshow(get_sample(31, result), cmap='gray')
        axes2[ind_i, ind_j].axis('off')
fig1.show()
fig1.savefig('noisy1_gaussian.png')
fig2.show()
fig2.savefig('noisy1_gaussian_sample.png')
# input('Press enter to continue...')

# %% Selection of preferred figures
fig1, axes1 = plt.subplots(1)
fig2, axes2 = plt.subplots(1)
fig1.suptitle('Best filtered result of noisy1 with a gaussian kernels')
axes1.imshow(my_cross_correlation(noisy1_grey, gaussian_kernel(9, 2), 'fill'), cmap='gray')
axes1.axis('off')
fig2.suptitle('Best filtered result of noisy2 with a gaussian kernels')
axes2.imshow(my_cross_correlation(noisy2_grey, gaussian_kernel(7, 4), 'fill'), cmap='gray')
axes2.axis('off')
fig1.show()
fig1.savefig('noisy1_best.png')
# input('Press enter to continue...')
fig2.show()
fig2.savefig('noisy2_best.png')


# input('Press enter to continue...')

# %% Biomedical problem


# Full documentation can be found on the first delivery.
# Taken from https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
def MyNormCCorrelation_201630945_0(image, kernel, boundary_condition):
    valid_conditions = ['fill', 'valid', 'symm']
    if boundary_condition not in valid_conditions:
        raise Exception('The boundary condition isn\'t valid (must be \'fill\', \'valid\' or \'symm\').')
    fill = 0
    img = np.asarray(image)
    ker = np.asarray(kernel)
    M, N = np.size(img, 0), np.size(img, 1)
    m, n = np.size(ker, 0), np.size(ker, 1)
    normalized_ccorrelation = np.zeros([M, N])
    a, b = int((m - 1) / 2), int((n - 1) / 2)
    # Template square sum is the same for each computation.
    t_sum = np.sum(np.square(kernel))
    for y in range(M):
        for x in range(N):
            s = 0
            i_sum = 0
            j = y - a
            while j < y + a + 1:
                i = x - b
                while i < x + b + 1:
                    if boundary_condition == 'valid' and (j < 0 or j >= M or i < 0 or i >= N):
                        s = img[y, x]
                        i_sum = img[y, x] ** 2
                        j = y + a
                        break
                    elif boundary_condition == 'fill' and (j < 0 or j >= M or i < 0 or i >= N):
                        s += fill * ker[j - y + a, i - x + b]
                        i_sum += (fill * ker[j - y + a, i - x + b]) ** 2
                    elif j < 0:
                        if boundary_condition == 'symm':
                            if i <= 0:
                                s += img[0, 0] * ker[j - y + a, i - x + b]
                                i_sum += (img[0, 0] * ker[j - y + a, i - x + b]) ** 2
                            elif i >= N - 1:
                                s += img[0, N - 1] * ker[j - y + a, i - x + b]
                                i_sum += (img[0, N - 1] * ker[j - y + a, i - x + b]) ** 2
                            else:
                                s += img[0, i] * ker[j - y + a, i - x + b]
                                i_sum += (img[0, i] * ker[j - y + a, i - x + b]) ** 2
                    elif j >= M:
                        if boundary_condition == 'symm':
                            if i <= 0:
                                s += img[M - 1, 0] * ker[j - y + a, i - x + b]
                                i_sum += (img[M - 1, 0] * ker[j - y + a, i - x + b]) ** 2
                            elif i >= N - 1:
                                s += img[M - 1, N - 1] * ker[j - y + a, i - x + b]
                                i_sum += (img[M - 1, N - 1] * ker[j - y + a, i - x + b]) ** 2
                            else:
                                s += img[M - 1, i] * ker[j - y + a, i - x + b]
                                i_sum += (img[M - 1, i] * ker[j - y + a, i - x + b]) ** 2
                    else:
                        if i < 0:
                            if boundary_condition == 'symm':
                                s += img[j, 0] * ker[j - y + a, i - x + b]
                                i_sum += (img[j, 0] * ker[j - y + a, i - x + b]) ** 2
                        elif i >= N:
                            if boundary_condition == 'symm':
                                s += img[j, N - 1] * ker[j - y + a, i - x + b]
                                i_sum += (img[j, N - 1] * ker[j - y + a, i - x + b]) ** 2
                        else:
                            s += img[j, i] * ker[j - y + a, i - x + b]
                            i_sum += (img[j, i] * ker[j - y + a, i - x + b]) ** 2
                    i += 1
                j += 1
            normalized_ccorrelation[y, x] = s / np.sqrt(t_sum * i_sum)
    return


# %% Template matching analysis
template_base = io.imread(os.path.join('template', 'reference1.jpg'))
template = rgb2gray(template_base)
# Manually selected kernels. Rescaling to
d1, y1, x1 = 53, 16, 214
kernel_1 = np.asarray(template[y1:(y1+d1), x1:(x1+d1)] * 255)
kernel_1 = kernel_1.astype(np.uint8)
d2, y2, x2 = 37, 54, 53
kernel_2 = np.asarray(template[y2:(y2+d2), x2:(x2+d2)] * 255)
kernel_2 = kernel_2.astype(np.uint8)
io.imsave('kernel1.png', kernel_1)
io.imsave('kernel2.png', kernel_2)

#%% Kernel plotting
fig, axes = plt.subplots(1, 2)
fig.suptitle('Kernels seleccionados para la detección de parásitos')
axes[0].imshow(kernel_1, cmap='gray')
axes[0].set_title('Kernel 1')
axes[0].axis('off')
axes[1].imshow(kernel_2, cmap='gray')
axes[1].set_title('Kernel 2')
axes[1].axis('off')
fig.show()
fig.savefig('kernels.png')
