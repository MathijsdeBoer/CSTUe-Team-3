# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 09:30:15 2017

@author: pmoeskops
"""

'''
Adding UNet for this given template
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
keras.backend.set_image_data_format('channels_last')
import random
random.seed(0)
import glob
import PIL.Image
import copy
from scipy.ndimage import rotate

from image_patch_functions import *
from Unet_architecture import *
from segmentation_prediction import make_predictions

#inputs
impaths_all = glob.glob(r'.\training\images\*.tif')
trainingsetsize = 15
patchsize = 32
minibatchsize = 200
minibatches = 10000


#shuffle the images to take a random subset for training later
random.shuffle(impaths_all)

maskpaths_all = copy.deepcopy(impaths_all)
segpaths_all = copy.deepcopy(impaths_all)

#select the corresponding masks and segmentations
for i in range(len(impaths_all)):
    maskpaths_all[i] = impaths_all[i].replace('images','mask')
    maskpaths_all[i] = maskpaths_all[i].replace('.tif','_mask.gif')

    segpaths_all[i] = impaths_all[i].replace('images','1st_manual')
    segpaths_all[i] = segpaths_all[i].replace('training.tif','manual1.gif')

print(impaths_all)
print(maskpaths_all)
print(segpaths_all)

#select the first 15 images as training set, the other 5 will be used for validation
impaths = impaths_all[:trainingsetsize]
maskpaths = maskpaths_all[:trainingsetsize]
segpaths = segpaths_all[:trainingsetsize]

#load the training images
images, masks, segmentations = loadImages(impaths,maskpaths,segpaths)

print(images.shape)
print(masks.shape)
print(segmentations.shape)

#pad the images with zeros to allow patch extraction at all locations
halfsize = int(patchsize/2)
images = np.pad(images,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
masks = np.pad(masks,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
segmentations = np.pad(segmentations,((0,0),(halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

#separately select the positive samples (vessel) and negative samples (background)
positivesamples = np.nonzero(segmentations)
negativesamples = np.nonzero(masks-segmentations)

print(len(positivesamples[0]))
print(len(negativesamples[0]))

trainnetwork = True

#initialise the network
cnn = Unet(pretrained_weights = 0)
#and start training
if trainnetwork:
    losslist = []

    for i in range(minibatches):

        posbatch = random.sample(list(range(len(positivesamples[0]))),int(minibatchsize/2))
        negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))

        Xpos, Ypos = make2Dpatches(positivesamples,posbatch,images,2*patchsize,1) # double patchsize for rotation
        Xneg, Yneg = make2Dpatches(negativesamples,negbatch,images,2*patchsize,0)   # it is cropped later

        # Data augmentation: Only rotation between 0 360 deg
        # For every patch, it creates 10 more rotated patches
        # Notice: that the minibatchsize becomes 11 times bigger!
        augmentations = 10
        Xpos_aug = copy.deepcopy(Xpos[:,halfsize:-halfsize,halfsize:-halfsize,:])
        Xneg_aug = copy.deepcopy(Xneg[:,halfsize:-halfsize,halfsize:-halfsize,:])
        Ypos_aug = copy.deepcopy(Ypos)
        Yneg_aug = copy.deepcopy(Yneg)

        for j in range(augmentations):
            angle = np.random.randint(361) # same angle for all samples
            Xpos_rot = rotate(Xpos, angle=angle, axes=(1,2), reshape=False)
            Xpos_rot = Xpos_rot[:,halfsize:-halfsize,halfsize:-halfsize,:]

            Xneg_rot = rotate(Xneg, angle=angle, axes=(1,2), reshape=False)
            Xneg_rot = Xneg_rot[:,halfsize:-halfsize,halfsize:-halfsize,:]

            Xpos_aug = np.vstack((Xpos_aug, Xpos_rot))
            Xneg_aug = np.vstack((Xneg_aug, Xneg_rot))

            Ypos_aug = np.vstack((Ypos_aug, Ypos))
            Yneg_aug = np.vstack((Yneg_aug, Yneg))

        Xtrain = np.vstack((Xpos_aug,Xneg_aug))
        Ytrain = np.vstack((Ypos_aug,Yneg_aug))

        loss = cnn.train_on_batch(Xtrain,Ytrain)
        losslist.append(loss)
        print('Batch: {}'.format(i))
        print('Loss: {}'.format(loss))


    plt.close('all')
    plt.figure()
    plt.plot(losslist)

    cnn.save(r'.\sub07_Unet.h5')

else:
    cnn = keras.models.load_model(r'.\sub07_Unet.h5')
    # cnn = keras.models.load_model(r'.\experiments\Unet10000.h5') # for debugging


#### Use the trained network to predict ####
# Paths to images/masks
valimpaths = impaths_all[trainingsetsize:]
valmaskpaths = maskpaths_all[trainingsetsize:]

testimpaths = glob.glob(r'.\test\images\*.tif')
testmaskpaths = glob.glob(r'.\test\mask\*.gif')

trainimpaths = impaths_all[:trainingsetsize]
trainmaskpaths = maskpaths_all[:trainingsetsize]

# Create directory to store results
dirName = "sub07_results"
try:
    # Create target directory
    os.mkdir(dirName)
    print("Directory ", dirName, " was created")
except FileExistsError:
    print("Directory", dirName, " already exists")
os.mkdir(dirName + "//training_results")
os.mkdir(dirName + "//validation_results")
os.mkdir(dirName + "//test_results")

debug = False # keep it False - othewise it does not use the trained network to predict
make_predictions(valimpaths, valmaskpaths, dirName, mode='val', cnn=cnn, halfsize=halfsize, debug=debug)
make_predictions(testimpaths, testmaskpaths, dirName, mode='test', cnn=cnn, halfsize=halfsize, debug=debug)
make_predictions(trainimpaths, trainmaskpaths, dirName, mode='train', cnn=cnn, halfsize=halfsize, debug=debug)
