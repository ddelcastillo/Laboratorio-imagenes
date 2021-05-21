import os
import glob
import joblib
import numpy as np
from skimage import io
from sklearn import svm
from skimage import color
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat
from functions import JointColorHistogram, CatColorHistogram
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def calculate_descriptors(data, parameters, calculate_dict):
    descriptor_matrix = np.zeros([len(data), parameters['texton_k']])
    if calculate_dict:
        filters_bank = loadmat(os.path.join('data_mp4', 'filterbank.mat'))['filterbank']
        filters_list = [filters_bank[:, :, i] for i in range(np.shape(filters_bank)[2])]
        gray_images = list(map(color.rgb2gray, data))
        calculateTextonDictionary_201622695_201630945(gray_images, filters_list, parameters)
    else:
        centroids = loadmat(parameters['dict_name'])['centroids']
        for i, img in enumerate(data):
            img_gray = color.rgb2gray(img)
            resp = calculateTextonHistogram_201622695_201630945(img_gray, centroids)
            descriptor_matrix[i, :] = resp
    return descriptor_matrix if not calculate_dict else None


def train(parameters, action):
    data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
    images_train = list(map(io.imread, glob.glob(data_train)))
    if action == 'save':
        # Saves the training descriptors into the 'train_descriptor_name' specified .npy file.
        calculate_descriptors(images_train, parameters, True)
        descriptors = calculate_descriptors(images_train, parameters, False)
        np.save(parameters['train_descriptor_name'], descriptors)
    else:
        pass  # Temporary dummy code for indentation.
    # Loads descriptors matrix.
    descriptors = np.load(parameters['train_descriptor_name'])
    y_hat, labels = [], set()
    class_to_number = {}
    for index, file in enumerate(os.listdir(os.path.join('data_mp4', 'scene_dataset', 'train'))):
        img_class = file.split('_')[0]
        if img_class not in labels:
            class_to_number[img_class] = len(labels)
            y_hat.append(len(labels))
            labels.add(img_class)
        else:
            y_hat.append(class_to_number[img_class])
    parameters['class_to_number']: class_to_number
    seed = 42  # Seed for deterministic random centroid generation.
    # model = KMeans(n_clusters=parameters['k'], random_state=seed)
    model = svm.SVC(kernel=parameters['kernel'])
    model.fit(descriptors, y_hat)
    # Persisting the model.
    joblib.dump(model, parameters['name_model'])


def validate(parameters, action):
    data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
    images_val = list(map(io.imread, glob.glob(data_val)))
    if action == 'load':
        descriptors = np.load(parameters['val_descriptor_name'])
    else:
        descriptors = calculate_descriptors(images_val, parameters, False)
        if action == 'save':
            # Saves the validation descriptors into the 'val_descriptor_name' specified .npy file.
            np.save(parameters['val_descriptor_name'], descriptors)
    # Loading model.
    model = joblib.load(parameters['name_model'])
    y_hat = model.predict(descriptors)
    # Obtaining the real cluster values for the validation labels.
    y_real = []
    class_to_number = parameters['class_to_number']
    for index, file in enumerate(os.listdir(os.path.join('data_mp4', 'scene_dataset', 'val'))):
        y_real.append(class_to_number[file.split('_')[0]])
    setting, y_real = 'micro', np.asarray(y_real)
    parameters['setting'] = setting
    conf_mat = confusion_matrix(y_real, y_hat)
    precision = precision_score(y_real, y_hat, average=setting)
    recall = recall_score(y_real, y_hat, average=setting)
    f_score = f1_score(y_real, y_hat, average=setting)
    return conf_mat, precision, recall, f_score


def print_results(conf_mat, precision, recall, f_score, parameters):
    # Confusion matrix print formatting (right justified).
    n = np.size(conf_mat, 0)
    col_separator, row_separator = '|', '—'  # Row separator should be a single character.
    cluster_size, padding = len(str(n)), 1
    len_col_sep, len_row_sep = len(col_separator), len(row_separator)
    cell_size = max(len(str(np.max(conf_mat))), cluster_size) + padding
    row_sep = ' ' * abs(cell_size - len(str(n)) + len_row_sep) + (
            (cell_size + len_col_sep) * n + len_col_sep) * row_separator

    title = ' Confusion matrix '
    print(row_separator * int(max(0, np.floor(len(row_sep) - len(title)) / 2)) + title + row_separator * int(
        np.ceil(max(0, len(row_sep) - len(title)) / 2)))
    print(row_sep)
    column_str = 'k' + ' ' * (cell_size - 1)  # Column headers.
    for i in range(n):
        column_str += ' ' * len_col_sep + ' ' * abs(cell_size - len(str(i))) + str(i)
    print(column_str)
    print(row_sep)
    # Printing each row with the values in the matrix.
    for i in range(np.size(conf_mat, 0)):
        row_str = str(i) + ' ' * abs(cell_size - len(str(i))) + col_separator
        for j in range(np.size(conf_mat, 1)):
            row_str += ' ' * abs(cell_size - len(str(conf_mat[i, j]))) + str(conf_mat[i, j]) + col_separator
        print(row_str)
    print(row_sep)
    print(f"Precision score ({parameters['setting']}): {'{:.3f}'.format(precision)}.")
    print(f"Recall score ({parameters['setting']}): {'{:.3f}'.format(recall)}.")
    print(f"F score ({parameters['setting']}): {'{:.3f}'.format(f_score)}.")
    print(row_sep)


def main(parameters, perform_train, action):
    if perform_train:
        train(parameters, action=action)
    conf_mat, precision, recall, f_score = validate(parameters, action=action)
    print_results(conf_mat, precision, recall, f_score, parameters)


# Runs each KMeans with the given seed and tests for [1, k_max] and plots the result
# to determine the optimal value for k to use for the texton dictionary result.
def _elbow_rule(data, seed, max_k):
    resp = []
    for k in range(1, max_k + 1):
        print(f"Iteration {k}/{max_k}.")
        model = KMeans(n_clusters=k, random_state=seed)
        model.fit(data)
        resp.append(model.inertia_)
    fig, axs = plt.subplots()
    fig.suptitle('Suma total de distancias cuadradas al centroide más\n cercano vs. número de centroides (k)')
    axs.plot(range(1, max_k + 1), resp, 'bo-')
    axs.set_ylabel("Distorción")
    axs.set_xlabel("k")
    axs.grid("on")
    fig.show()
    input("Press enter to continue...")


# Calculates the filter response through cross-correlation of the image with the list of filters.
# Flatten the result and stores it in a (m x n) x #filters matrix (one row per pixel, each column a filter response).
def calculateFilterResponse_201622695_201630945(img_gray, filters):
    assert img_gray.ndim == 2, f"Image must be a gray image (2D, found {img_gray.ndim} dimensions)."
    resp = np.zeros([img_gray.size, len(filters)])
    for i, f in enumerate(filters):
        resp[:, i] = correlate2d(img_gray, f, mode='same').flatten()
    return resp


# Calculates the texton dictionary for the given list of images with the given list of filters.
# It stores the texton dictionary with respect to the assigned parameters name.
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
    savemat(parameters['dict_name'], texton_dict)


# Finds the normalized histogram for the image's response to the filters given the centroids dictionary.
# Returns the normalized texton histogram for the image (with respect to filters) with the given centroids.
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


if __name__ == '__main__':
    parameters = {
                  'name_model': 'final_model_201622695_201630945.joblib',
                  'train_descriptor_name': 'DDC_IM_train_descriptor.npy',
                  'val_descriptor_name': 'DDC_IM_val_descriptor.npy',
                  # Based on the best result. Will be overwritten with training.
                  'label_clusters': {'buildings': {'cluster': 1, 'count': 2},
                                     'glacier': {'cluster': 2, 'count': 2},
                                     'mountains': {'cluster': 7, 'count': 2},
                                     'street': {'cluster': 3, 'count': 5},
                                     'forest': {'cluster': 4, 'count': 4},
                                     'sea': {'cluster': 5, 'count': 1}},
                  # Based on testing for texton dictionary.
                  'texton_k': 6,
                  'dict_name': 'textons_model_201622695_201630945.mat',
                  'kernel': 'rbf',
                  'class_to_number': {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountains': 3, 'sea': 4, 'street': 5}
                  }

    perform_train = False
    action = 'none'
    main(parameters=parameters, perform_train=perform_train, action=action)
