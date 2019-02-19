# Adapted from Team Challenge Team 4 code

import glob
import numpy as np
import os
import random
import SimpleITK as sitk

from os import listdir
from os.path import isfile, join

def mutate_data(image_patches, label_patches, noise_strength=20, new_records=1000, seed=314159):
    count = 0
    random.seed(seed)

    print("Mutating Data...")
    while count < new_records:
        # Fetch random patient images
        print("{}/{}".format(count, new_records), end='\r')

        idx = random.randint(0, image_patches.shape[0])

        image_patch = image_patches[idx,:,:,:]
        label_patch = label_patches[idx,:,:,:]

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
                np.rot90(image_patch)

                np.rot90(label_patch)
            elif rot == 1:
                # Rotate twice
                np.rot90(image_patch)
                np.rot90(image_patch)

                np.rot90(label_patch)
                np.rot90(label_patch)
            else:
                # Rotate thrice
                np.rot90(image_patch)
                np.rot90(image_patch)
                np.rot90(image_patch)

                np.rot90(label_patch)
                np.rot90(label_patch)
                np.rot90(label_patch)

        # Random noise (normal distribution)
        elif operation == 1:
            # Noise is only applied to the actual image
            image_patch = np.random.normal(image_patch, noise_strength)

        # Save
        np.append(image_patches, image_patch, axis=0)
        np.append(label_patches, label_patch, axis=0)

        count += 1

    print("Mutated images")