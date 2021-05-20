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
