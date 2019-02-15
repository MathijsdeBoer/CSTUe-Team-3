# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 09:30:15 2017

@author: pmoeskops
"""

'''
Adding UNet for this given templete
'''
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

def buildLeNet():
    
    cnn = keras.models.Sequential()
    
    layer0 = keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1))
    cnn.add(layer0)
    print(layer0.input_shape)
    print(layer0.output_shape)
    
    layer1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    cnn.add(layer1)
    print(layer1.output_shape)
    
    layer2 = keras.layers.Conv2D(16, (5, 5), activation='relu')
    cnn.add(layer2)
    print(layer2.output_shape)
    
    layer3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    cnn.add(layer3)
    print(layer3.output_shape)
    
    layer4 = keras.layers.Flatten() 
    cnn.add(layer4)
    print(layer4.output_shape)
    
    layer5 = keras.layers.Dense(120, activation='relu')
    cnn.add(layer5)
    print(layer5.output_shape)
    
    layer6 = keras.layers.Dense(84, activation='relu')
    cnn.add(layer6)
    print(layer6.output_shape)
    
    layer7 = keras.layers.Dense(2, activation='softmax')
    cnn.add(layer7)
    print(layer7.output_shape)
    
    adam = keras.optimizers.adam(lr=0.001)
    cnn.compile(loss='categorical_crossentropy', optimizer=adam)
    
    return cnn

def Unet(pretrained_weights = None):
    input_size=(32, 32, 1)
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
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

    cnn.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

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
        #First we do the RGB conversion
        red_channal = np.array(PIL.Image.open(impaths[i]),dtype=np.int16)[:,:,0]
        green_channal = np.array(PIL.Image.open(impaths[i]),dtype=np.int16)[:,:,1]
        blue_channal = np.array(PIL.Image.open(impaths[i]),dtype=np.int16)[:,:,2]
        image = 0.299 * red_channal + 0.587 * green_channal + 0.114 * blue_channal
        #Then the image is normalized to 0-1
        min_max_scaler = preprocessing.MinMaxScaler()
        image = min_max_scaler.fit_transform(image)
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

    
def main():
    
    #inputs
    impaths_all = glob.glob(r'C:\Users\lxs\Desktop\Q3\medical imaging\project1\training\images\*.tif') 
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
    #cnn = buildLeNet()
    cnn = Unet()
    #and start training
    if trainnetwork: 
        losslist = []
        
        for i in range(minibatches):
            
            posbatch = random.sample(list(range(len(positivesamples[0]))),int(minibatchsize/2))
            negbatch = random.sample(list(range(len(negativesamples[0]))),int(minibatchsize/2))
             
            Xpos, Ypos = make2Dpatches(positivesamples,posbatch,images,32,1)
            Xneg, Yneg = make2Dpatches(negativesamples,negbatch,images,32,0)
          
            Xtrain = np.vstack((Xpos,Xneg))
            Ytrain = np.vstack((Ypos,Yneg))            
            
            loss = cnn.train_on_batch(Xtrain,Ytrain)
            losslist.append(loss)
            print('Batch: {}'.format(i))
            print('Loss: {}'.format(loss))
                
        
        plt.close('all')
        plt.figure()
        plt.plot(losslist)    
        
        cnn.save(r'C:\Users\lxs\Desktop\Q3\medical imaging\project1\result\sample\trainednetwork.h5')
    
    else:
        cnn = keras.models.load_model(r'C:\Users\lxs\Desktop\Q3\medical imaging\project1\result\sample\trainednetwork.h5')
    
    #validate the trained network on the 5 images that were left out during training (numbers 15 to 19)       
    valimpaths = impaths_all[trainingsetsize:]
    valmaskpaths = maskpaths_all[trainingsetsize:]
    
    for j in range(len(valimpaths)):   
        print(valimpaths[j])
        
        valimage = np.array(PIL.Image.open(valimpaths[j]),dtype=np.int16)[:,:,1]
        valmask = np.array(PIL.Image.open(valmaskpaths[j]),dtype=np.int16)
        
        valimage = np.pad(valimage,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
        valmask = np.pad(valmask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
        
        valsamples = np.nonzero(valmask)
        
        probimage = np.zeros(valimage.shape)
        
        probabilities = np.empty((0,))
        
        minibatchsize = 100 #can be as large as memory allows during testing
        
        for i in range(0,len(valsamples[0]),minibatchsize):
            print('{}/{} samples labelled'.format(i,len(valsamples[0])))
            
            if i+minibatchsize < len(valsamples[0]):
                valbatch = np.arange(i,i+minibatchsize)        
            else:
                valbatch = np.arange(i,len(valsamples[0]))        
            
            Xval = make2Dpatchestest(valsamples,valbatch,valimage,patchsize)
                    
            prob = cnn.predict(Xval, batch_size=minibatchsize)
            probabilities = np.concatenate((probabilities,prob[:,1]))     
          
        for i in range(len(valsamples[0])):
            probimage[valsamples[0][i],valsamples[1][i]] = probabilities[i]    
            
        plt.figure()
        plt.imshow(probimage,cmap='Greys_r')
        plt.axis('off')

    return   
     
    
if __name__=="__main__":
    main()