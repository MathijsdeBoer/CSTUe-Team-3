import glob
import numpy as np
import os
import random
import SimpleITK as sitk

from os import listdir
from os.path import isfile, join

# Change if needed
noise_strength = 20
data_path = r"C:\Users\czori\Documents\Team Challenge Data\TeamChallenge\Data"
new_records = 10

# Do not change
count = 0
random.seed(314159)

def get_image4D_from_array(array):
# Horrible, Horrible Hack
    tdim = array.shape[0]
    slices = []
    for t in range(tdim):
        slices.append( sitk.GetImageFromArray( array[t,:,:,:], False ) )
    img = sitk.JoinSeries(slices)
    return img

print("Mutating Data...")
# Fetches underlying paths
paths = [x[0] for x in os.walk(data_path)]
while count < new_records:
    # Fetch random patient images
    path = paths[random.randint(0, len(paths))]
    print("{}/{}".format(count, new_records), end='\r')

    # Make sure we don't try to process the root folder
    if path.endswith("Data"):
        continue

    # Load images
    image_4d = sitk.ReadImage(join(path, "patient{}_4d.nii.gz".format(path[-3:])))

    # ED
    image_frameED = sitk.ReadImage(join(path, "patient{}_frame01.nii.gz".format(path[-3:])))
    image_frameED_gt = sitk.ReadImage(join(path, "patient{}_frame01_gt.nii.gz".format(path[-3:])))

    image_frameES = None
    image_frameES_gt = None

    ES_frame = ""
    ES_path = glob.glob(join(path, "patient{}_frame*.nii.gz".format(path[-3:])))
    ES_gt_path = glob.glob(join(path, "patient{}_frame*_gt.nii.gz".format(path[-3:])))

    # ES
    for ES in ES_path:
        if ES != join(path, "patient{}_frame01.nii.gz".format(path[-3:])) and "_gt" not in ES:
            image_frameES = sitk.ReadImage(ES)
            ES_frame = ES[-9:-7]
            break

    for ES in ES_gt_path:
        if ES != join(path, "patient{}_frame01_gt.nii.gz".format(path[-3:])) and "_gt" in ES:
            image_frameES_gt = sitk.ReadImage(ES)
            break


    # Mutate images
    # Determine which operation to perform
    operation = random.randint(0, 1)

    image_4d_res = None
    image_frameED_res = None
    image_frameED_gt_res = None
    image_frameES_res = None
    image_frameES_gt_res = None

    # Random Rotation
    if operation == 0:
        np_image_4d = sitk.GetArrayFromImage(image_4d)
        np_image_frameED = sitk.GetArrayFromImage(image_frameED)
        np_image_frameED_gt = sitk.GetArrayFromImage(image_frameED_gt)
        np_image_frameES = sitk.GetArrayFromImage(image_frameES)
        np_image_frameES_gt = sitk.GetArrayFromImage(image_frameES_gt)

        # Determine new orientation
        rot = random.randint(0, 2)

        # There's probably a cleaner way to do this, but it works
        if rot == 0:
            # Rotate once
            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)
        elif rot == 1:
            # Rotate twice
            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)

            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)
        else:
            # Rotate thrice
            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)
            
            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)

            np.rot90(np_image_4d)
            np.rot90(np_image_frameED)
            np.rot90(np_image_frameED_gt)
            np.rot90(np_image_frameES)
            np.rot90(np_image_frameES_gt)

        image_4d_res = get_image4D_from_array(np_image_4d)
        image_frameED_res = sitk.GetImageFromArray(np_image_frameED)
        image_frameED_gt_res = sitk.GetImageFromArray(np_image_frameED_gt)
        image_frameES_res = sitk.GetImageFromArray(np_image_frameES)
        image_frameES_gt_res = sitk.GetImageFromArray(np_image_frameES_gt)

    # Random noise (normal distribution)
    elif operation == 1:
        np_image_4d = sitk.GetArrayFromImage(image_4d)
        np_image_frameED = sitk.GetArrayFromImage(image_frameED)
        np_image_frameES = sitk.GetArrayFromImage(image_frameES)

        np_image_4d = np.random.normal(np_image_4d, noise_strength)
        np_image_frameED = np.random.normal(np_image_frameED, noise_strength)
        np_image_frameES = np.random.normal(np_image_frameES, noise_strength)

        image_4d_res = get_image4D_from_array(np_image_4d)
        image_frameED_res = sitk.GetImageFromArray(np_image_frameED)
        image_frameED_gt_res = image_frameED_gt
        image_frameES_res = sitk.GetImageFromArray(np_image_frameES)
        image_frameES_gt_res = image_frameES_gt

    # Make sure path exists that we can write to
    if not os.path.exists(join(data_path, "MUT", "Patient{}".format(count + 101))):
        os.makedirs(join(data_path, "MUT", "Patient{}".format(count + 101)))

    # Copy original metadata
    image_4d_res.CopyInformation(image_4d)
    image_frameED_res.CopyInformation(image_frameED)
    image_frameED_gt_res.CopyInformation(image_frameED_gt)
    image_frameES_res.CopyInformation(image_frameES)
    image_frameES_gt_res.CopyInformation(image_frameES_gt)

    # Write images
    sitk.WriteImage(image_4d_res, join(data_path, "MUT", "Patient{}".format(count + 101), "patient{}_4d.nii.gz".format(count + 101)))
    sitk.WriteImage(image_frameED_res, join(data_path, "MUT", "Patient{}".format(count + 101), "patient{}_frame01.nii.gz".format(count + 101)))
    sitk.WriteImage(image_frameED_gt_res, join(data_path, "MUT", "Patient{}".format(count + 101), "patient{}_frame01_gt.nii.gz".format(count + 101)))
    sitk.WriteImage(image_frameES_res, join(data_path, "MUT", "Patient{}".format(count + 101), "patient{}_frame{}.nii.gz".format(count + 101, ES_frame)))
    sitk.WriteImage(image_frameES_gt_res, join(data_path, "MUT", "Patient{}".format(count + 101), "patient{}_frame{}_gt.nii.gz".format(count + 101, ES_frame)))
    
    count += 1

print("Mutated images")