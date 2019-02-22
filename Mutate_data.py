# Adapted from Team Challenge Team 4 code

import glob
import numpy as np
import os
import random
import SimpleITK as sitk

from os import listdir
from os.path import isfile, join

def mutate_data(images, masks, segmentations, noise_strength=20, new_records=1000, seed=314159, gradient_min=0.7, gradient_max=1.0):
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
    print("\tGradient Min = {}".format(gradient_min))
    print("\tGradient Max = {}".format(gradient_max))
    while count < new_records:
        # Fetch random patient images
        print("{}/{}".format(count, new_records), end='\r')

        idx = random.randint(0, images.shape[0] - 1)

        image = images[idx]
        mask = masks[idx]
        segmentation = segmentations[idx]

        # Mutate images
        # Determine which operation to perform
        operation = random.randint(0, 2)

        # Random Rotation
        if operation == 3:
            # Determine new orientation
            rot = random.randint(1, 3)
            image = np.rot90(image, k=rot)
            mask = np.rot90(mask, k=rot)
            segmentation = np.rot90(segmentation, k=rot)

        # Random noise (normal distribution)
        elif operation == 1:
            # Noise is only applied to the actual image
            image = np.random.normal(image, noise_strength)

        # Add gradient
        elif operation == 2:
            # Generate gradient and give it a random orientation
            gradient = np.zeros(image.shape)
            length = image.shape[0]
            for i in range(length):
                gradient[i] = gradient_min + (gradient_max - gradient_min) * (i / length)
            rot = random.randint(0, 1)
            if rot == 1:
                gradient = np.rot90(gradient, k=2)

            # Apply gradient
            image = image * gradient

        # Save
        res_im = np.append(res_im, [image], axis=0)
        res_ma = np.append(res_ma, [mask], axis=0)
        res_se = np.append(res_se, [segmentation], axis=0)

        count += 1

    print("Mutated images")
    return res_im, res_ma, res_se