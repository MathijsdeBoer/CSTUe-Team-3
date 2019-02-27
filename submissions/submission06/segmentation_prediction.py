import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from image_patch_functions import *

def make_predictions(impaths, maskpaths, dirName, mode, cnn, halfsize=32, debug=False, minibatchsize=1000):
  """
  Produces the segmentation probability arrays and the corresponding
  PNG images. Note: predictions need to be thresholded before submitted

  mode: 'train', 'val' or 'test'
  cnn: the trained model
  """

  for j in range(len(impaths)):
      print(impaths[j])

      # Keep only green channel. Note that the scalling takes place in the paches
      image = np.array(PIL.Image.open(impaths[j]),dtype=np.int16)[:,:,1]
      mask = np.array(PIL.Image.open(maskpaths[j]),dtype=np.int16)

      image = np.pad(image,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)
      mask = np.pad(mask,((halfsize,halfsize),(halfsize,halfsize)),'constant', constant_values=0)

      samples = np.nonzero(mask)
      probimage = np.zeros(image.shape)
      probabilities = np.empty((0,))

      for i in range(0,len(samples[0]),minibatchsize):
          print('{}/{} samples labelled'.format(i,len(samples[0])))

          if i+minibatchsize < len(samples[0]):
              batch = np.arange(i,i+minibatchsize)
          else:
              batch = np.arange(i,len(samples[0]))

          X = make2Dpatchestest(samples,batch,image,patchsize=2*halfsize)

          if debug:
              prob = np.random.rand(batch.shape[0], 2) # used for debugging
          else:
              prob = cnn.predict(X, batch_size=minibatchsize)

          probabilities = np.concatenate((probabilities,prob[:,1]))

      for i in range(len(samples[0])):
          probimage[samples[0][i],samples[1][i]] = probabilities[i]

      # Save the predictions
      if mode=="train":
          foldername = "//training_results//"
      elif mode=="val":
          foldername = "//validation_results//"
      elif mode=="test":
          foldername = "//test_results//"

      path_prob = dirName + foldername + mode + "_probabilities_{}".format(j+1)
      np.save(path_prob, probimage)

      path_img = dirName + foldername + mode + "{}.png".format(j+1)

      plt.figure()
      plt.imshow(probimage,cmap='Greys_r')
      plt.axis('off')
      plt.savefig(path_img)

  return
