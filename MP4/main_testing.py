# %%
import os
import glob
import joblib
import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import JointColorHistogram
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# %%
parameters = {'histogram_function': JointColorHistogram,
              'space': 'HSV', 'transform_color_function': color.rgb2hsv,  # Esto es solo un ejemplo.
              'bins': 6, 'k': 6,
              'name_model': 'model.joblib',
              'train_descriptor_name': 'DDC_IM_train_descriptor.npy',
              'val_descriptor_name': 'DDC_IM_val_descriptor.npy'}

perform_train = True  # Cambiar parámetro a False al momento de hacer la entrega
action = 'save'  # Cambiar a None al momento de hacer la entrega

# %%
data_train = os.path.join('data_mp4', 'scene_dataset', 'train', '*.jpg')
images_train = list(map(io.imread, glob.glob(data_train)))


# %%
def calculate_descriptors(data, parameters):
    if parameters['space'] != 'RGB':
        data = list(map(parameters['transform_color_function'], data))
    bins = [parameters['bins']] * len(data)
    histograms = list(map(parameters['histogram_function'], data, bins))
    descriptor_matrix = np.array(histograms)
    descriptor_matrix = descriptor_matrix.reshape((len(data), np.prod(descriptor_matrix.shape[1:])))
    return descriptor_matrix


# %% Test
if action == 'save':
    descriptors = calculate_descriptors(images_train, parameters)
    # Saves the training descriptors into the 'train_descriptor_name' specified .npy file.
    np.save(parameters['train_descriptor_name'], descriptors)
else:
    # Esta condición solo la tendrán que utilizar para la tercera entrega.
    pass  # Temporary dummy code for indentation.
# Loads descriptors matrix.

# %%
descriptors = np.load(parameters['train_descriptor_name'])
seed = 42  # Seed for deterministic random centroid generation.
model = KMeans(n_clusters=parameters['k'], random_state=seed)
model.fit(descriptors)
# Persisting the model.
joblib.dump(model, parameters['name_model'])

# %%
y_hat = model.predict(descriptors)
clusters, labels = set(np.unique(y_hat)), set()
cluster_labels = {element: {} for element in clusters}
class_counter = {}
for index, file in enumerate(os.listdir(os.path.join('data_mp4', 'scene_dataset', 'train'))):
    img_class = file.split('_')[0]
    if img_class not in labels:
        labels.add(img_class)
        class_counter[img_class] = 1
        for cluster in cluster_labels:
            cluster_labels[cluster][img_class] = 0
    else:
        cluster_labels[y_hat[index]][img_class] += 1
        class_counter[img_class] += 1
# Ordering the assigned labels in ascending order for each cluster.
for cluster in cluster_labels:
    cluster_labels[cluster] = dict(sorted(cluster_labels[cluster].items(), key=lambda item: item[1], reverse=True))


# %%
def assign_labels(p_label_clusters, p_cluster_labels, p_cluster, p_label=None):
    finished, i = False, 0
    label_counts = list(p_cluster_labels[p_cluster].keys())
    while not finished:
        label, count = label_counts[i], p_cluster_labels[p_cluster][label_counts[i]]
        if label not in p_label_clusters:
            p_label_clusters[label], finished = {'cluster': p_cluster, 'count': count}, True
        else:
            if p_label is None or (p_label is not None and p_label != label):
                if count > p_label_clusters[label]['count']:
                    temp = p_label_clusters[label]
                    p_label_clusters[label], finished = {'cluster': p_cluster, 'count': count}, True
                    # Re-assigns the newly assigned cluster
                    assign_labels(p_label_clusters, p_cluster_labels, temp['cluster'])
                else:
                    i += 1
            else:
                i += 1
        if i == len(label_counts):
            break


label_clusters = {}
for cluster in cluster_labels:
    assign_labels(label_clusters, cluster_labels, cluster)
parameters['label_clusters'] = label_clusters

# %%

fig, axs = plt.subplots(len(class_counter), max(class_counter.values()))
title = 'Cluster asignado a cada imagen por clase (una clase por fila)'
fig.suptitle(title)
classes = list(class_counter.keys())
for i in range(len(class_counter)):
    for j, image in enumerate(glob.glob(os.path.join('data_mp4', 'scene_dataset', 'train', classes[i] + '*.jpg'))):
        axs[i, j].imshow(io.imread(image))
        axs[i, j].axis('off')
        axs[i, j].set_title(y_hat[i * len(class_counter) + j])
fig.tight_layout(pad=1)
fig.show()
fig.savefig('test_figure.png')

# %% Validate

data_val = os.path.join('data_mp4', 'scene_dataset', 'val', '*.jpg')
images_val = list(map(io.imread, glob.glob(data_val)))
conf_mat, precision, recall, f_score = 0, 0, 0, 0
if action == 'load':
    descriptors = np.load(parameters['val_descriptor_name'])
    # Esta condición solo la tendrán que utilizar para la tercera entrega.
else:
    descriptors = calculate_descriptors(images_val, parameters)
    if action == 'save':
        # Saves the validation descriptors into the 'val_descriptor_name' specified .npy file.
        np.save(parameters['val_descriptor_name'], descriptors)
        # Loading model.
        model = joblib.load(parameters['name_model'])
        y_hat = model.predict(descriptors)

        # Obtaining the real cluster values for the validation labels.
        y_real = []
        label_clusters = parameters['label_clusters']
        for index, file in enumerate(os.listdir(os.path.join('data_mp4', 'scene_dataset', 'val'))):
            img_class = file.split('_')[0]
            y_real.append(label_clusters[img_class]['cluster'])
        setting = 'micro'
        parameters['setting'] = setting
        conf_mat = confusion_matrix(y_real, y_hat)
        precision = precision_score(y_real, y_hat, average=setting)
        recall = recall_score(y_real, y_hat, average=setting)
        f_score = f1_score(y_real, y_hat, average=setting)
# return conf_mat, precision, recall, f_score

# %%
# Confusion matrix print formatting (right justified).
col_separator, row_separator = '|', '—'  # Row separator should be a single character.
cluster_size, padding = len(str(parameters['k'])), 1
len_col_sep, len_row_sep = len(col_separator), len(row_separator)
cell_size = max(len(str(np.max(conf_mat))), cluster_size) + padding
row_sep = ' ' * abs(cell_size - len(str(parameters['k'])) + len_row_sep) + (
        (cell_size + len_col_sep) * parameters['k'] + len_col_sep) * row_separator

title = ' Confusion matrix '
print(row_separator * int(max(0, np.floor(len(row_sep) - len(title)) / 2)) + title + row_separator * int(
    np.ceil(max(0, len(row_sep) - len(title)) / 2)))
print(row_sep)
column_str = 'k' + ' ' * (cell_size - 1)  # Column headers.
for i in range(parameters['k']):
    column_str += ' ' * len_col_sep + ' ' * abs(cell_size - len(str(i))) + str(i)
print(column_str)
print(row_sep)
# Printing each row with the values in the matrix.
for i in range(parameters['k']):
    row_str = str(i) + ' ' * abs(cell_size - len(str(i))) + col_separator
    for j in range(parameters['k']):
        row_str += ' ' * abs(cell_size - len(str(conf_mat[i, j]))) + str(conf_mat[i, j]) + col_separator
    print(row_str)
print(row_sep)
print(f"Precision score ({parameters['setting']}): {precision}.")
print(f"Recall score ({parameters['setting']}): {recall}.")
print(f"F score ({parameters['setting']}): {f_score}.")
print(row_sep)
print(f"Color transformation function: {str(parameters['transform_color_function']).split(' ')[1]}")
print(f"Histogram function: {str(parameters['histogram_function']).split(' ')[1]}")
print(f"Number of cluster (k): {parameters['k']}")
print(f"Number of bins: {parameters['bins']}")
print(f"Color space: {parameters['space']}")
# Incluyan los parámetros que usaron y las métricas de validación.
