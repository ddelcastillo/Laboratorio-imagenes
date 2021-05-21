import os
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat


def calculateFilterResponse_201622695_201630945(img_gray, filters):
    assert img_gray.ndim == 2, f"Image must be a gray image (2D, found {img_gray.ndim} dimensions)."
    resp = np.zeros([img_gray.size, len(filters)])
    for i, f in enumerate(filters):
        resp[:, i] = correlate2d(img_gray, f, mode='same').flatten()
    return resp


def calculateTextonDictionary_201622695_201630945(images_train, filters, parameters):
    if len(images_train) == 0:
        return np.zeros(0)
    # Code relies on concatenation instead of a pre-set sized matrix to allow flexible image sizes.
    resp = np.zeros([0, len(filters)])
    for img in images_train:
        resp = np.concatenate((resp, calculateFilterResponse_201622695_201630945(img, filters)))
    # K-means
    seed = 42  # Seed for deterministic random centroid generation.
    texton_model = KMeans(n_clusters=parameters['texton_k'], random_state=seed)
    texton_model.fit(resp)
    texton_dict = {'centroids': texton_model.cluster_centers_}
    savemat(parameters['dictname'], texton_dict)


# Finds the normalized histogram for the image's response to the filters given the centroids dictionary.
def calculateTextonHistogram_201622695_201630945(img_gray, centroids):
    assert img_gray.ndim == 2, f"Image must be a gray image (2D, found {img_gray.ndim} dimensions)."
    n, m = np.size(img_gray, 0), np.size(img_gray, 1)
    filters_bank = loadmat(os.path.join('data_mp4', 'filterbank.mat'))['filterbank']
    filters_list = [filters_bank[:, :, i] for i in range(np.shape(filters_bank)[2])]
    filter_response = calculateFilterResponse_201622695_201630945(img_gray, filters_list).reshape(
        (n, m, len(filters_list)))
    image_centroids = np.zeros([n, m])
    # Goes through each pixel and finds the centroid closer to the pixel's filter response.
    # It assigns the centroid, then calculated the histogram (one bin per texton).
    for i in range(n):
        for j in range(m):
            centroid, distance = -1, np.inf
            for ind, k in enumerate(centroids):
                temp_distance = np.linalg.norm(filter_response[i, j, :] - k)
                if temp_distance < distance:
                    centroid, distance = ind, temp_distance
            image_centroids[i, j] = centroid
    hist = np.histogram(image_centroids, bins=len(centroids))
    return hist[0] / np.sum(hist[0])  # Normalized histogram

