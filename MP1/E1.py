# %% Imports
import numpy as np
import math
import requests
import skimage.io as io
import shutil
import os
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# %% Part 1 and 2
URL = "https://web.stanford.edu/class/ee368/Handouts/Lectures/Examples/11-Edge-Detection/Hough_Transform_Circles" \
      "/coins.png"
r = requests.get(URL)
with open('coins.png', 'wb') as f:
    f.write(r.content)
f.close()

img = io.imread('coins.png')
print(f'Las dimensiones de la imagen son {img.shape} pixeles.')
print(f'El tipo de datos de la imagen es {img.dtype}.')

# %% Part 3 and 4
plt.style.use("bmh")
fig, axes = plt.subplots(1, 2)
ax = axes.ravel()
fig.suptitle('Comparación del número de bins')
ax[0].set_title('256 bins', size=12, pad=10)
ax[0].set_xlabel('Intensidad del pixel (-)')
ax[0].set_ylabel('Frecuencia')
ax[0].set_xlim(0, 255)
ax[0].grid(False)
ax[1].set_title('32 bins', size=12, pad=10)
ax[1].set_xlabel('Intensidad del pixel (-)')
ax[1].set_ylabel('Frecuencia')
ax[1].set_xlim(0, 255)
ax[1].grid(False)
ax[0].hist(img.flatten(), bins=256)
ax[1].hist(img.flatten(), bins=32)
fig.show()
# input("Press Enter to continue...")

#%% Part 6
# Taken from documentation: https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
thresh = threshold_otsu(img)    # Otsu treshold.
binary = img > thresh           # Image binarization.
print(f'El valor del umbral para la imagen es {thresh}.')
fig, ax = plt.subplots()
ax.imshow(binary, cmap=plt.cm.gray)
ax.set_title('Imagen binaria con un umbral de Otsu')
ax.axis('off')
fig.show()
fig.savefig('coins_binary_1.png')

#%% Part 7
thresh_2 = np.percentile(img, 60)   # New threshold is 60th percentile of the image.
binary_2 = img > thresh_2
print(f'El valor del umbral con el percentil 60 es {thresh_2}.')
fig, ax = plt.subplots()
ax.imshow(binary_2, cmap=plt.cm.gray)
ax.set_title('Imagen binaria con un umbral del percentil 60')
ax.axis('off')
fig.show()
fig.savefig('coins_binary_2.png')

#%% Part 8
thresh_3 = 175
binary_3 = img > thresh_3
fig, ax = plt.subplots()
ax.imshow(binary_3, cmap=plt.cm.gray)
ax.set_title('Imagen binaria con un umbral de 175')
ax.axis('off')
fig.show()
fig.savefig('coins_binary_3.png')


