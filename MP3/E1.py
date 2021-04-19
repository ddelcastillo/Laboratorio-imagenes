# %% Packages
import os
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.filters import median, threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import disk

# %% Image pre-processing
# The number of images to show in the subplot (N > 1).
n = 3
i = 0
fig, axes = plt.subplots(n, 2)
filtered_images = []
for file in os.listdir(os.path.join('.', 'noisy_data')):
    # Main loop for processing each image in the database.
    img = io.imread(os.path.join('.', 'noisy_data', file))
    # The image is filtered with a median filtered, then converted to
    # grayscale, and then the Otsu threshold is used to binary the image.
    filtered_img = median(img)
    gray_filtered_img = rgb2gray(filtered_img)
    threshold = threshold_otsu(gray_filtered_img)
    binary_img = gray_filtered_img < threshold
    # Removing the larger white cells through an aperture operation.
    r = 14  # Arbitrary value found through experimentation.
    dilated_binary_img = binary_dilation(binary_erosion(binary_img, disk(r)), disk(2 * r))
    dilated_img_inverse = np.logical_not(dilated_binary_img).astype('int')
    intersection_img = binary_img & dilated_img_inverse
    if i < n:
        # For an upper title divided by each column of images.
        if i == 0:
            axes[i, 0].set_title('Imágenes originales')
            axes[i, 1].set_title('Imágenes preprocesadas')
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(intersection_img, cmap='gray')
        axes[i, 1].axis('off')
        i += 1
    filtered_images.append(intersection_img)
fig.show()
fig.savefig('preprocess_sample.png')


# input('Press enter to continue...')

# %% Hole filling algorithm

def MyHoleFiller_201630945_201622695(bin_img):
    # Creation of the f marker:
    n, m = np.size(bin_img, 0), np.size(bin_img, 1)
    f = np.zeros([n, m], dtype='int')
    r = min(2, min(n * 2 + 1, m * 2 + 1))
    b = disk(r)
    for i in range(n):
        for j in range(m):
            if (i == 0 or i == (n - 1)) and (j == 0 or j == (m - 1)):
                f[i, j] = 1 - bin_img[i, j]
    finish = False
    h = f
    g = np.logical_not(bin_img).astype('int')
    while not finish:
        temp = h
        h = binary_dilation(h, b) & g
        if (temp == h).all():
            finish = True
    return np.logical_not(h).astype('int')


# %% Testing
# Commented code for the test of the algorithm with the star image.
# star_img = io.imread('star_binary.png')
# full_star = MyHoleFiller_201630945_201622695(star_img)
# plt.imshow(full_star, cmap='gray')
# plt.show()

# %% Iterating over truth files and comparing:
i = 0
for truth_file in os.listdir(os.path.join('.', 'groundtruth')):
    truth_img = io.imread(os.path.join('.', 'groundtruth', truth_file))
    full_img = MyHoleFiller_201630945_201622695(filtered_images[i])
    inter1 = np.sum(truth_img & filtered_images[i])
    union1 = np.sum(truth_img | filtered_images[i])
    inter2 = np.sum(truth_img & full_img)
    union2 = np.sum(truth_img | full_img)
    print(f'Img {i + 1}: | Preprocesamiento: {inter1 / union1} | Preprocesamiento + huecos: {inter2 / union2}')
    i += 1
