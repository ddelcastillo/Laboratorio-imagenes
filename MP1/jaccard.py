import skimage.io as io
import numpy as np
import scipy.io
import os

from skimage.filters import threshold_otsu


# Calculates the Jaccard index for two binary or boolean Numpy arrays.
def jaccard_index(annotation, mask):
    # Makes sure that annotation and mask are binary (in case they're boolean).
    t_annotation = 1 * annotation
    t_mask = 1 * mask
    # Intersection: element-wise AND operator between binary matrixes.
    # Union: element-wise OR operator between binary matrixes.
    # The magnitude of each set is the sum of it's elements (binary).
    inter = np.sum(t_annotation & t_mask)
    union = np.sum(t_annotation | t_mask)
    return inter / union


# %% Jaccard index tests for the coins example.
img = io.imread('coins.png')  # Coins image.
m = scipy.io.loadmat(os.path.join('coins_gt.mat'))
main = m['gt']  # Image annotation.

otsu_thresh = threshold_otsu(img)  # Otsu threshold.
perc_thresh = np.percentile(img, 60)  # 60th percentile of the image threshold.

# Binary masks for each threshold.
otsu = img > otsu_thresh
perc = img > perc_thresh

print(f'El índice de Jaccard para el umbral de Otsu es {jaccard_index(main, otsu)}.')
print(f'El índice de Jaccard para el umbral del percentil 60 es {jaccard_index(main, perc)}.')
