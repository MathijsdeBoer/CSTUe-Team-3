import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from skimage.draw import circle
from skimage.morphology import medial_axis

thickness_threshold = 3

# Load
img = Image.open(r"D:\UU\CSTUe\training\1st_manual\21_manual1.gif")
arr = np.array(img)

# Skeletonize and find distances through the Medial Axis Transform
skeleton, distance = medial_axis(arr, return_distance=True)
med_axis = skeleton * distance

# Split in thin and thick
thin_mat = np.array(med_axis)
thin_mat[thin_mat > thickness_threshold] = 0
thick_mat = np.array(med_axis)
thick_mat[thick_mat <= thickness_threshold] = 0

thin_vessels = np.zeros(arr.shape)
thick_vessels = np.zeros(arr.shape)

# Draw Circles
for coord in np.argwhere(thin_mat > 0):
    rr, cc = circle(coord[0], coord[1], thin_mat[coord[0], coord[1]], shape=thin_vessels.shape)
    thin_vessels[rr, cc] = 1

for coord in np.argwhere(thick_mat > 0):
    rr, cc = circle(coord[0], coord[1], thick_mat[coord[0], coord[1]], shape=thin_vessels.shape)
    thick_vessels[rr, cc] = 1

# Write
thin_image = Image.fromarray((thin_vessels * 255).astype(np.uint8))
thin_image.save(r"D:\UU\CSTUe\training\1st_manual\thin\21_manual1.gif")
thick_image = Image.fromarray((thick_vessels * 255).astype(np.uint8))
thick_image.save(r"D:\UU\CSTUe\training\1st_manual\thick\21_manual1.gif")