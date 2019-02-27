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
from sklearn import preprocessing

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras import backend as keras

def Unet(pretrained_weights = None):
    input_size=(32, 32, 1)

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',trainable = 0)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = Conv2D(1, 1,activation='softmax')(conv9)

    conv11 = Flatten()(conv9)
    conv11 = Dense(120, activation='relu')(conv11)

    conv12 = Dense(84, activation='relu')(conv11)

    cov13 = Dense(2, activation='softmax')(conv12)

    cnn = Model(input = inputs, output = cov13)

    cnn.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
        print('Loading the pretrained weights!')
        cnn.load_weights(r'.\Uet_transfer_withtop.h5')
    return cnn

def make2Dpatches(samples, batch, images, patchsize, label):

    halfsize = int(patchsize/2)

    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
    Y = np.zeros((len(batch),2),dtype=np.int16)

    for i in range(len(batch)):

        patch = images[samples[0][batch[i]],(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize),(samples[2][batch[i]]-halfsize):(samples[2][batch[i]]+halfsize)]

        X[i,:,:,0] = patch
        Y[i,label] = 1

    return X, Y


def make2Dpatchestest(samples, batch, image, patchsize):

    halfsize = int(patchsize/2)

    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)

    for i in range(len(batch)):

        patch = image[(samples[0][batch[i]]-halfsize):(samples[0][batch[i]]+halfsize),(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize)]

        X[i,:,:,0] = patch

    return X

def loadImages(impaths,maskpaths,segpaths):

    images = []
    masks = []
    segmentations = []

    for i in range(len(impaths)):
        # Keep only green channel. Note that the scalling takes place in the paches
        image = np.array(PIL.Image.open(impaths[i]),dtype=np.int16)[:,:,1]
        #Load masks and segmentation
        mask = np.array(PIL.Image.open(maskpaths[i]),dtype=np.int16)
        segmentation = np.array(PIL.Image.open(segpaths[i]),dtype=np.int16)

        images.append(image)
        masks.append(mask)
        segmentations.append(segmentation)

    images = np.array(images)
    masks = np.array(masks)
    segmentations = np.array(segmentations)


    return images, masks, segmentations


#inputs
impaths_all = glob.glob(r'.\training\images\*.tif')
trainingsetsize = 15
patchsize = 32
minibatchsize = 200
minibatches = 2000


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
cnn = Unet(pretrained_weights = 1)
#and start training
if trainnetwork:
    losslist = []

    for i in range(minibatches):

        posbatch = random.sample(list(range(len(positivesamples[0]))),int(minibatchsize/2))
        negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))

        Xpos, Ypos = make2Dpatches(positivesamples,posbatch,images,32,1)
        Xneg, Yneg = make2Dpatches(negativesamples,negbatch,images,32,0)

        # Z-score normalization (Global Contrast Normalization)
        # Calculate mean per patch
        mean_per_patch_pos = np.mean(Xpos, axis=(1,2), keepdims=True)
        mean_per_patch_neg = np.mean(Xneg, axis=(1,2), keepdims=True)

        # Calculate std per patch
        std_per_patch_pos = np.std(Xpos, axis=(1,2), keepdims=True)
        std_per_patch_neg = np.std(Xneg, axis=(1,2), keepdims=True)

        # Scale
        Xpos = (Xpos - mean_per_patch_pos) / std_per_patch_pos
        Xneg = (Xneg - mean_per_patch_neg) / std_per_patch_neg

        Xtrain = np.vstack((Xpos,Xneg))
        Ytrain = np.vstack((Ypos,Yneg))

        loss = cnn.train_on_batch(Xtrain,Ytrain)
        losslist.append(loss)
        print('Batch: {}'.format(i))
        print('Loss: {}'.format(loss))


    plt.close('all')
    plt.figure()
    plt.plot(losslist)

    cnn.save(r'.\neofrombegining2000.h5')

else:
    cnn = keras.models.load_model(r'.\sub04_Unet.h5')

#validate the trained network on the 5 images that were left out during training (numbers 15 to 19)
valimpaths = impaths_all[trainingsetsize:]
valmaskpaths = maskpaths_all[trainingsetsize:]

# Create directory to store results
dirName = "sub04_results"
try:
    # Create target directory
    os.mkdir(dirName)
    print("Directory ", dirName, " was created")
except FileExistsError:
    print("Directory", dirName, " already exists")
os.mkdir(dirName + "//training_results")
os.mkdir(dirName + "//validation_results")
os.mkdir(dirName + "//test_results")

#################################################################################
# EVALUATE FOR THE VALIDATION SET
for j in range(len(valimpaths)):
    print(valimpaths[j])

    # Keep only green channel. Note that the scalling takes place in the paches
    valimage = np.array(PIL.Image.open(valimpaths[j]),dtype=np.int16)[:,:,1]
    valmask = np.array(PIL.Image.open(valmaskpaths[j]),dtype=np.int16)

    valimage = np.pad(valimage,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    valmask = np.pad(valmask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

    valsamples = np.nonzero(valmask)

    probimage = np.zeros(valimage.shape)

    probabilities = np.empty((0,))

    minibatchsize = 1000 #can be as large as memory allows during testing

    for i in range(0,len(valsamples[0]),minibatchsize):
        print('{}/{} samples labelled'.format(i,len(valsamples[0])))

        if i+minibatchsize < len(valsamples[0]):
            valbatch = np.arange(i,i+minibatchsize)
        else:
            valbatch = np.arange(i,len(valsamples[0]))

        Xval = make2Dpatchestest(valsamples,valbatch,valimage,patchsize)
        # Z-score normalization (Global Contrast Normalization)
        # Calculate mean per patch
        mean_per_patch_val = np.mean(Xval, axis=(1,2), keepdims=True)

        # Calculate std per patch
        std_per_patch_val = np.std(Xval, axis=(1,2), keepdims=True)

        # Scale
        Xval = (Xval - mean_per_patch_val) / std_per_patch_val

        prob = cnn.predict(Xval, batch_size=minibatchsize)
        # prob = np.random.rand(valbatch.shape[0], 2) # used for debugging
        probabilities = np.concatenate((probabilities,prob[:,1]))

    for i in range(len(valsamples[0])):
        probimage[valsamples[0][i],valsamples[1][i]] = probabilities[i]

    val_path_prob = dirName + "//validation_results//" + "val_probabilities_{}".format(j+1)
    np.save(val_path_prob, probimage)

    val_path_img = dirName + "//validation_results//" + "{}.png".format(j+1)

    plt.figure()
    plt.imshow(probimage,cmap='Greys_r')
    plt.axis('off')
    plt.savefig(val_path_img)

################################################################################
# Repeat for TEST SET
test_paths = glob.glob(r'.\test\images\*.tif')
test_maskpaths = copy.deepcopy(impaths_all)

#select the corresponding masks and segmentations
for i in range(len(test_paths)):
    test_maskpaths[i] = test_paths[i].replace('images','mask')
    test_maskpaths[i] = test_maskpaths[i].replace('.tif','_mask.gif')

for j in range(len(test_paths)):
    print(test_paths[j])

    # Keep only green channel. Note that the scalling takes place in the paches
    testimage = np.array(PIL.Image.open(test_paths[j]),dtype=np.int16)[:,:,1]
    testmask = np.array(PIL.Image.open(test_maskpaths[j]),dtype=np.int16)

    testimage = np.pad(testimage,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    testmask = np.pad(testmask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

    testsamples = np.nonzero(testmask)

    probimage = np.zeros(testimage.shape)

    probabilities = np.empty((0,))

    minibatchsize = 1000 #can be as large as memory allows during testing

    for i in range(0,len(testsamples[0]),minibatchsize):
        print('{}/{} test samples labelled'.format(i,len(testsamples[0])))

        if i+minibatchsize < len(testsamples[0]):
            testbatch = np.arange(i,i+minibatchsize)
        else:
            testbatch = np.arange(i,len(testsamples[0]))

        Xtest = make2Dpatchestest(testsamples,testbatch,testimage,patchsize)
        # Z-score normalization (Global Contrast Normalization)
        # Calculate mean per patch
        mean_per_patch_test = np.mean(Xtest, axis=(1,2), keepdims=True)

        # Calculate std per patch
        std_per_patch_test = np.std(Xtest, axis=(1,2), keepdims=True)

        # Scale
        Xtest = (Xtest - mean_per_patch_test) / std_per_patch_test

        prob = cnn.predict(Xtest, batch_size=minibatchsize)
        # prob = np.random.rand(testbatch.shape[0], 2) # used for debugging
        probabilities = np.concatenate((probabilities,prob[:,1]))

    for i in range(len(testsamples[0])):
        probimage[testsamples[0][i],testsamples[1][i]] = probabilities[i]
    # Save probabilities for each pixel
    test_path_prob = dirName + "//test_results//" + "test_probabilities_{}".format(j+1)
    np.save(test_path_prob, probimage)

    test_path_img = dirName + "//test_results//" + "{}.png".format(j+1)

    plt.figure()
    plt.imshow(probimage,cmap='Greys_r')
    plt.axis('off')
    plt.savefig(test_path_img)

################################################################################
# EVALUATE FOR THE TRAINING SET
for j in range(len(impaths)):
    print(impaths[j])

    # Keep only green channel. Note that the scalling takes place in the paches
    image = np.array(PIL.Image.open(impaths[j]),dtype=np.int16)[:,:,1]
    mask = np.array(PIL.Image.open(maskpaths[j]),dtype=np.int16)

    image = np.pad(image,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
    mask = np.pad(mask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

    trainsamples = np.nonzero(mask)

    probimage = np.zeros(image.shape)

    probabilities = np.empty((0,))

    minibatchsize = 1000 #can be as large as memory allows during testing

    for i in range(0,len(trainsamples[0]),minibatchsize):
        print('{}/{} samples labelled'.format(i,len(trainsamples[0])))

        if i+minibatchsize < len(trainsamples[0]):
            trainbatch = np.arange(i,i+minibatchsize)
        else:
            trainbatch = np.arange(i,len(trainsamples[0]))

        Xtrain = make2Dpatchestest(trainsamples,trainbatch,image,patchsize)
        # Z-score normalization (Global Contrast Normalization)
        # Calculate mean per patch
        mean_per_patch_train = np.mean(Xtrain, axis=(1,2), keepdims=True)

        # Calculate std per patch
        std_per_patch_train = np.std(Xtrain, axis=(1,2), keepdims=True)

        # Scale
        Xtrain = (Xtrain - mean_per_patch_train) / std_per_patch_train

        prob = cnn.predict(Xtrain, batch_size=minibatchsize)
        # prob = np.random.rand(trainbatch.shape[0], 2) # used for debugging
        probabilities = np.concatenate((probabilities,prob[:,1]))

    for i in range(len(trainsamples[0])):
        probimage[trainsamples[0][i],trainsamples[1][i]] = probabilities[i]

    train_path_prob = dirName + "//training_results//" + "train_probabilities_{}".format(j+1)
    np.save(train_path_prob, probimage)

    train_path_img = dirName + "//training_results//" + "{}.png".format(j+1)

    plt.figure()
    plt.imshow(probimage,cmap='Greys_r')
    plt.axis('off')
    plt.savefig(train_path_img)
