# Adapted from Team Challenge Team 4 code

import glob
import numpy as np
import os
import random
import SimpleITK as sitk

from os import listdir
from os.path import isfile, join

def mutate_data(images, masks, segmentations, noise_strength=20, new_records=1000, seed=314159):
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
        operation = random.randint(0, 1)

        # Random Rotation
        if operation == 0:

            # Determine new orientation
            rot = random.randint(0, 2)

            # There's probably a cleaner way to do this, but it works
            if rot == 0:
                # Rotate once
                np.rot90(image)

                np.rot90(mask)

                np.rot90(segmentation)
            elif rot == 1:
                # Rotate twice
                np.rot90(image)
                np.rot90(image)

                np.rot90(mask)
                np.rot90(mask)

                np.rot90(segmentation)
                np.rot90(segmentation)
            else:
                # Rotate thrice
                np.rot90(image)
                np.rot90(image)
                np.rot90(image)

                np.rot90(mask)
                np.rot90(mask)
                np.rot90(mask)

                np.rot90(segmentation)
                np.rot90(segmentation)
                np.rot90(segmentation)

        # Random noise (normal distribution)
        elif operation == 1:
            # Noise is only applied to the actual image
            image = np.random.normal(image, noise_strength)

        # Save
        res_im = np.append(res_im, [image], axis=0)
        res_ma = np.append(res_ma, [mask], axis=0)
        res_se = np.append(res_se, [segmentation], axis=0)

        count += 1

    print("Mutated images")
    return res_im, res_ma, res_se