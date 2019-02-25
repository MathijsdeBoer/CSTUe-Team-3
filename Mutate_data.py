# Adapted from Team Challenge Team 4 code

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

    res_im = np.array(images)
    res_ma = np.array(masks)
    res_se = np.array(segmentations)

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
        res_im = np.append(res_im, [image], axis=0)
        res_ma = np.append(res_ma, [mask], axis=0)
        res_se = np.append(res_se, [segmentation], axis=0)

        count += 1

    print("Mutated images")
    return res_im, res_ma, res_se
