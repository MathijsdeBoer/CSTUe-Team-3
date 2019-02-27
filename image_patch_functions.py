import numpy as np
import PIL.Image

def make2Dpatches(samples, batch, images, patchsize, label):

    halfsize = int(patchsize/2)

    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)
    Y = np.zeros((len(batch),2),dtype=np.int16)

    for i in range(len(batch)):

        patch = images[samples[0][batch[i]],(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize),(samples[2][batch[i]]-halfsize):(samples[2][batch[i]]+halfsize)]

        X[i,:,:,0] = patch
        Y[i,label] = 1

    # Z-score normalization (Global Contrast Normalization)
    # Calculate mean per patch
    mean_per_patch = np.mean(X, axis=(1,2), keepdims=True)

    # Calculate std per patch
    std_per_patch = np.std(X, axis=(1,2), keepdims=True)

    # Scale
    X = (X - mean_per_patch) / std_per_patch

    return X, Y


def make2Dpatchestest(samples, batch, image, patchsize):

    halfsize = int(patchsize/2)

    X = np.empty([len(batch),patchsize,patchsize,1],dtype=np.float32)

    for i in range(len(batch)):

        patch = image[(samples[0][batch[i]]-halfsize):(samples[0][batch[i]]+halfsize),(samples[1][batch[i]]-halfsize):(samples[1][batch[i]]+halfsize)]

        X[i,:,:,0] = patch

    # Z-score normalization (Global Contrast Normalization)
    # Calculate mean per patch
    mean_per_patch = np.mean(X, axis=(1,2), keepdims=True)

    # Calculate std per patch
    std_per_patch = np.std(X, axis=(1,2), keepdims=True)

    # Scale
    X = (X - mean_per_patch) / std_per_patch
    
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
