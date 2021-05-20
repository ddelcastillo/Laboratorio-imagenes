# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from scipy.signal import correlate2d
from scipy.io import loadmat, savemat
from skimage import color
import joblib
import glob
import os


# TODO Copiar y pegar estas funciones en el script principal (main_Codigo1_Codigo2.py)
# TODO Cambiar el nombre de las funciones para incluir sus códigos de estudiante

# %%
def calculateFilterResponse_201622695_201630945(img_gray, filters):
    assert img_gray.ndim == 2, f"Image must be a gray image (2D, found {img_gray.ndim} dimensions)."
    resp = np.zeros([img_gray.size, len(filters)])
    for i, f in enumerate(filters):
        resp[:, i] = correlate2d(img_gray, f, mode='same').flatten()
    return resp


# %%
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
    return resp


# %%
def _elbow_rule(data, seed, max_k):
    resp = []
    for k in range(1, max_k+1):
        print(f"Iteration {k}/{max_k}...")
        model = KMeans(n_clusters=k, random_state=seed)
        model.fit(data)
        resp.append(model.inertia_)
    fig, axs = plt.subplots()
    fig.suptitle('Suma total de distancias cuadradas al centroide más\n cercano vs. número de centroides (k)')
    axs.plot(range(1, max_k+1), resp, 'bo-')
    axs.set_ylabel("Distorción")
    axs.set_xlabel("k")
    axs.grid("on")
    fig.show()
    # input("Press enter to continue...")
    return resp

# %%
data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
images_train = list(map(color.rgb2gray, list(map(io.imread, glob.glob(data_train)))))

filters_bank = loadmat(os.path.join('data_mp4', 'filterbank.mat'))['filterbank']
filters_list = [filters_bank[:, :, i] for i in range(np.shape(filters_bank)[2])]

# %%
test_image = color.rgb2gray(io.imread('data_mp4/scene_dataset/train/buildings_1.jpg'))
test_response = calculateFilterResponse_201622695_201630945(test_image, filters_list)
test_filter_response = calculateTextonDictionary_201622695_201630945([test_image], filters_list, {})
# %%
filter_response = calculateTextonDictionary_201622695_201630945(images_train, filters_list, {})
# TODO Borrar los comentarios marcados con un TODO.
# %% Saving data
np.save("train_pixels_cc.npy", filter_response)
# %% Getting results for elbow rule
inertia_values = _elbow_rule(filter_response, 42, 30)
# %%
np.save("inertia_results.npy", inertia_values)
# %% Visualization
fig, axs = plt.subplots()
axs.plot(range(1, 30), inertia_values, 'bo-')
axs.plot(6, inertia_values[5], 'ro')
axs.annotate(text="k = 6", xy=(6, inertia_values[5]), xytext=(40,20), textcoords='offset points', ha='center',
             va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5',
                             color='red'))
fig.suptitle('Suma total de distancias cuadradas al centroide más\n cercano vs. número de centroides (k)')
axs.grid("on")
axs.set_xlabel("k")
axs.set_ylabel("Distorción")
fig.show()
fig.savefig('elbow.png')
# %% Best model training
model = KMeans(n_clusters=6, random_state=42)
model.fit(filter_response)
# %% Save mat
savemat("textons_model.mat", {'centroids': model.cluster_centers_})
# %%
centroids = loadmat("textons_model.mat")['centroids']
# %% Testing elbow method call
inertia_values = _elbow_rule(filter_response, 42, 2)
