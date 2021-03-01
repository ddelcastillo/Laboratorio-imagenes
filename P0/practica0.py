import numpy as np
import requests
import skimage.io as io
import os
import matplotlib.pyplot as plt

from skimage.color import rgb2gray

# Fetching, saving and writing the giraffe image.
URL = 'https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/images/cc_Giraffes_16x9.jpg' \
      '?itok=dKmuVKO6 '
r = requests.get(URL)
with open('jirafas.jpg', 'wb') as f:
    f.write(r.content)  

# Graph of the RGB image and it's luminance image
fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.suptitle('RGB and Luminance of the Giraffe Image')
img = io.imread(os.path.join('', 'jirafas.jpg'))
img_gray = rgb2gray(img)
ax1.set_title('RGB')
ax2.set_title('Luminance of original RGB (no grayscale)', fontsize='8')
ax1.imshow(img)
ax2.imshow(img_gray)
fig1.show()
input("Press Enter to continue...")

# Graph of the RGB image and each of it's channels
print(f'The shape of the image is: {img.shape}.')

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle('RGB and Channels of the Giraffe Image')
ax1.set_title('RGB')
ax2.set_title('Channel 1 (Red)')
ax3.set_title('Channel 2 (Green)')
ax4.set_title('Channel 3 (Blue)')
ax1.imshow(img)
ax2.imshow(img[:, :, 0], cmap='gray')
ax3.imshow(img[:, :, 1], cmap='gray')
ax4.imshow(img[:, :, 2], cmap='gray')
fig2.show()
fig2.savefig('jirafas_canales.jpg')
input("Press Enter to continue...")

# Histogram of RGB picture intensity
fig3, ax = plt.subplots(1, 1)
fig3.suptitle('Histogram of Giraffe RGB Picture Intensity')
ax.hist(img.flatten(), bins=256)
ax.set_ylabel('Count')
ax.set_xlabel('Intensity')
fig3.show()
fig3.savefig('jirafas_histograma.jpg')
input("Press Enter to continue...")

# Histogram of RGB grayscale picture intensity
fig4, ax = plt.subplots(1, 1)
fig4.suptitle('Histogram of Giraffe RGB Grayscale Picture Intensity')
ax.hist(img_gray.flatten(), bins=256)
ax.set_ylabel('Count')
ax.set_xlabel('Intensity')
fig4.show()
fig4.savefig('jirafas_gris_histograma.jpg')
input("Press Enter to continue...")

fig5, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig5.suptitle('RGB and Greyscale Images and Histograms')
ax1.set_title('Color Image')
ax2.set_title('Color Image Histogram')
ax3.set_title('Grayscale Image')
ax4.set_title('Grayscale Image Histogram')
ax1.imshow(img)
ax2.hist(img.flatten(), bins=256)
ax3.imshow(img_gray, cmap='gray')
ax4.hist(img_gray.flatten(), bins=256)
fig5.show()
fig5.savefig('jirafas_imagenes_histogramas.jpg')


