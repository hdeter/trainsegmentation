# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:48:16 2022

@author: hdeter

Example pipeline for using trainablesegmenation
"""
import trainsegmentation as tseg
from skimage.io import imread
from matplotlib import pyplot as plt
from skimage import filters

#locations of images
imgdir = './single_cell_data'

#list of binary label dir -- one directory per label -- must be a list (even if its only one directory)
## labels should be binary masks with same shapes and names as corresponding images
### labels will be generated in the order they are provided!
maskdirs = [imgdir + '/cell-masks',imgdir + '/background-masks']

#load training data (images and labels)
#generates list of images and corresponding labels (each mask is given a number progressively to generate label (e.g. 1,2))
IMG, LABEL = tseg.import_training_data(imgdir,maskdirs,ext = '.tif')

#define training feature sets --> see documentation for details
featureselect = ['Sklearn_basic','Watershed_distance','Neighbors']
#get training features for given images and labels
##appends data to one large array to feed into classifier
###savedatafile -> name of file to save training data (uses pickle)
savedatafile = imgdir + '/training-data.pkl'
traininglabels, trainingfeatures, featureselect = tseg.get_training_data(IMG,LABEL,featureselect,savedatafile = savedatafile)
#import classifier
# traininglabels, trainingfeatures,featureselect = tseg.load_training_data(savedatafile)

#train the classifier
##you can initiate your own sklearn classifier and pass in clf // otherwise initiates RandomForestClassifier
###saveclftofile uses pickle to save classifier to file
saveclftofile = imgdir + '/classifier.pkl'
clf = tseg.train_classifier(traininglabels,trainingfeatures,featureselect, saveclftofile = saveclftofile, clf = None)
# clf = tseg.load_classifier(saveclftofile)

###now we have several options for applying classifier
##first load a test image
img = imread(imgdir + '/test000025xy1c1.tif')

#option 1: binary classifier -> doesn't work well for this training set
result1 = tseg.classify_image(img,clf,featureselect)
plt.imshow(result1)

#option 2: probablity classifier -> works but we want a binary mask of just one label
result2 = tseg.classify_image_probability(img,clf,featureselect)
plt.imshow(result2)

#option 3: binary classifier for one label
result3 = tseg.classify_image_label(img,clf,featureselect)
plt.imshow(result3)

#option 4: probability classifier for one label
result4 = tseg.classify_image_label_probability(img,clf,featureselect)
plt.imshow(result4)

#bonus option: add thresholding to probability mask to get binary mask
result5 = tseg.threshold_mask(result4, threshmethod = filters.threshold_isodata)
plt.imshow(result5)
