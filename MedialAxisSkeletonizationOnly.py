# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:44:26 2019

@author: s158000
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.viewer import ImageViewer
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.data import binary_blobs
from skimage.draw import circle
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage.viewer import ImageViewer
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.data import binary_blobs
import matplotlib.image as mpimg
from skimage.draw import circle
from PIL import Image

img=mpimg.imread('21_manual1.gif')

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(img, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, axes = plt.subplots(2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap='magma', interpolation='nearest')
ax[1].contour(img, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

fig.tight_layout()
plt.show()