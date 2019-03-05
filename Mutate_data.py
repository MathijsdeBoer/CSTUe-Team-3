'''
Adapted from Team Challenge Team 4 code, 05-03-2019

In this code, data is augmentated by mutating the original training set. 
There are 4 operations, which are randomly applied to the training images using random.randint(0,3). 
1) Rotation: The image, mask and segmentations of the trainingset are fliped over the x-axis, the y-axis or both. 
   The flip direction is randomly chosen again using random.randint(0,2); 
2) Noise: A normal distributed noise is only applied to the actual image; 
3) Contrast change: The pixel values of the image is multiplied with a random chosen integer between 0.5 and 1.5; 
4) Blur Kernel: Gaussian blur with sigma 2 is applied to the image. 
Input parameters: 
    images, masks, segmentations: the original of the training set. 
    noise_strength: integer, is used as standard deviation in the noise operation. 
    new_records: amount of generated training samples which will be added to the original. 
Output: 
    The original and generated images, masks and segmentations.
'''

import glob
import numpy as np
import os
import random
import scipy
import SimpleITK as sitk

from os import listdir
from os.path import isfile, join
from scipy import ndimage

def mutate_data(images, masks, segmentations, noise_strength=64, new_records=1000, seed=314159):
    count = 0
    random.seed(seed)

    # Get how many original images we have
    original_amount = images.shape[0]

    # Preallocate array space for new images
    new_shape = images.shape
    new_shape = list(new_shape)
    new_shape[0] += new_records
    new_shape = tuple(new_shape)

    res_im = np.zeros(new_shape)
    res_ma = np.zeros(new_shape)
    res_se = np.zeros(new_shape)

    # Place original images into result array
    res_im[0:images.shape[0], 0:images.shape[1], 0:images.shape[2]] = images
    res_ma[0:masks.shape[0], 0:masks.shape[1], 0:masks.shape[2]] = masks
    res_se[0:segmentations.shape[0], 0:segmentations.shape[1], 0:segmentations.shape[2]] = segmentations

    print("Mutating Data...")
    print("SETTINGS")
    print("\tNoise Strength = {}".format(noise_strength))
    print("\tNew Records = {}".format(new_records))
    print("\tSeed = {}".format(seed))
    while count < new_records:
        # Fetch random patient images
        print("{}/{}".format(count, new_records), end='\r')

        idx = random.randint(0, images.shape[0] - 1)

        image = images[idx]
        mask = masks[idx]
        segmentation = segmentations[idx]

        # Mutate images
        # Determine which operation to perform
        operation = random.randint(0, 3)

        # Random Rotation
        if operation == 0:
            # Determine new orientation
            rot = random.randint(0, 2)

            if rot == 0:
                # Flip over one axis
                image = np.flipud(image)
                mask = np.flipud(mask)
                segmentation = np.flipud(segmentation)
            elif rot ==1:
                # Flip over one axis
                image = np.fliplr(image)
                mask = np.fliplr(mask)
                segmentation = np.fliplr(segmentation)
            else:
                # Flip over both axes
                image = np.flipud(image)
                mask = np.flipud(mask)
                segmentation = np.flipud(segmentation)

                image = np.fliplr(image)
                mask = np.fliplr(mask)
                segmentation = np.fliplr(segmentation)

        # Random noise (normal distribution)
        elif operation == 1:
            # Noise is only applied to the actual image
            image = np.random.normal(image, noise_strength)
        # Contrast change
        elif operation == 2:
            image = (float(random.randint(50, 150)) / 100.0) * image
        # Blur Kernel
        elif operation == 3:
            image = ndimage.gaussian_filter(image, 2)

        # Save
        res_im[original_amount + count] = image
        res_ma[original_amount + count] = mask
        res_se[original_amount + count] = segmentation

        count += 1

    print("Mutated images")
    return res_im, res_ma, res_se
