# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:49:47 2022

@author: hdeter

Contains several methods to get image features and functions to train 
and classify images using sci-kit learn

"""

###########################


import numpy as np
import math
import os
from scipy import ndimage,signal
from scipy import ndimage as ndi

from skimage import (
     feature, filters
)
from skimage.io import imread

from sklearn.ensemble import RandomForestClassifier

import glob

import pickle

import itertools

#################################

#subfunctions for use in features

def set_kernels(kernel,a,b):
    kernel[a,b] = 1
    return kernel

def Basic_filter(img,minSigma,maxSigma,basic_func,name):
#applies a basic filter to pixels with shifted directionality with a distance of sigma
    #get stuff to store
    meta = []
    features = []
        
    sigmastep = 2
    sigma = minSigma/sigmastep

    while sigma < maxSigma:

        #loop through each sigma
        ##for each new key initiate empty array
        sigma = int(sigma*sigmastep)
        keyname = name + '_' + str(sigma)
        # print(sigma)
        
        #roll data through sigmas to get each pixel
        ranges = itertools.product(list(range(-sigma,sigma+1)),list(range(-sigma,sigma+1)))
        filtered = [np.roll(img,(x,y)) for x,y in ranges]
        
        filtered = np.dstack(filtered)
        data = basic_func(filtered,axis = -1)
                
        #store data then move to next key
        meta.append(keyname)
        features.append(data)
                
    features = np.dstack(features)
    
    return meta, features


#Functions for getting 2D features from images 
## all functions return list of metadat (description of each feature) and 3d arrays with features along axis = -1


#Neighbors
def Neighbors(img,minSigma = 1,maxSigma = 16):
    sigmastep = 2
    sigma = minSigma/sigmastep
    
    meta = []
    features = []
    
    while sigma <= maxSigma:
        sigma = int(sigma*sigmastep)
        # print(sigma)
                

        
        #loop through +/- for each sigma
        ##for each new key initiate empty array
        keyname = 'Neighbors_' + str(sigma) + '_%d'
        
        #create kernel to get neighbors
        kernel = np.zeros((sigma*2+1,sigma*2+1))
        kranges = itertools.product(range((-1*sigma), sigma+1,sigma),range((-1*sigma), sigma+1, sigma))
        kernels = [set_kernels(kernel,a,b) for a,b in kranges]
        neighbors = [ndimage.convolve(img,kernel,mode = 'mirror') for kernel in kernels]
        keys = [keyname % k for k in range(len(neighbors))]
                                                    
        #store data then move to next key
        meta = meta + keys
        features = features + neighbors

    features = np.dstack([f for f in features])
                        
    return meta, features    



#Membrane Projections
def Membrane_projections(img,nAngles = 30, patchSize = 19,membraneSize = 1):

    
    ##create 30 kernels to use for image convolution##
    
    #create initial kernel
    membranePatch = np.zeros((patchSize,patchSize))
    
    middle = int((patchSize-1)/2)
    startX = int(middle - math.floor(membraneSize/2))
    endX = int(middle + math.ceil(membraneSize/2))
    
    membranePatch[:,startX:endX] = 1.0
    
    #rotate kernels
    rotationAngle = 180/nAngles
    rotatedPatches = []
    
    
    for i in range(nAngles):
        rotatedPatch = np.copy(membranePatch)
        rotatedPatch = ndimage.rotate(rotatedPatch,rotationAngle*i,reshape = False)

        #convolve the image
        ##must flip kernel first when using filter2D?
        rotatedPatch = np.flip(rotatedPatch,-1)
        
        rp = signal.convolve2d(img,rotatedPatch,mode = 'same')
        rotatedPatches.append(rp)
    
    rotatedPatches = np.dstack(rotatedPatches)
     
    meta = []
    features = []
    
    methods = [np.sum,np.mean,np.std,np.median,np.max,np.min]
    
    for i,method in enumerate(methods):
        keyname = 'Membrane_projections_' + str(i) + '_' + str(patchSize) + '_' + str(membraneSize)
        projection = method(rotatedPatches,axis = 2)
                            
        meta.append(keyname)
        features.append(projection)
        
    features = np.dstack([f for f in features])
    
    return meta, features
    
def Gaussian_blur(img,minSigma = 1,maxSigma = 16):
    
    meta = []
    features = []
     
    sigmastep = 2
    sigma = minSigma/sigmastep
    while sigma < maxSigma:
        sigma = int(sigma*sigmastep)
     
        keyname = 'Gaussian_blur_' + str(sigma) + '.0'
        #make gaussian blur
        gblur = ndimage.gaussian_filter(img,0.4*sigma)
        
        meta.append(keyname)
        features.append(gblur)
        
    features = np.dstack([f for f in features])
         
    return meta, features

def Difference_of_Gaussians(img,minSigma = 1,maxSigma = 16):
    
    meta = []
    features = []
    
    lastimage = img
    lastsigma = 0
     
    sigmastep = 2
    sigma = minSigma/sigmastep
    while sigma < maxSigma:
        sigma = int(sigma*sigmastep)
     
        keyname = 'Difference_of_Gaussians_' + str(sigma) + str(lastsigma)
        #make gaussian blur
        gblur = ndimage.gaussian_filter(img,0.4*sigma)
        diff = np.subtract(gblur,lastimage)
        
        meta.append(keyname)
        features.append(diff)
        
        #store last image and sigma
        lastimage = gblur
        lastsigma = sigma
        
        
    features = np.dstack([f for f in features])
         
    return meta, features

def Sobel_filter(img,minSigma = 1,maxSigma = 16):
    #counter to track iterations
    scount = 0
    
    #no gaussian first
    keyname = 'Sobel_filter_%01d.0'  
    sfilter = ndimage.sobel(img)
    
    #store data
    meta = [keyname %scount]
    features = [sfilter]
    scount = scount+1
         
    sigmastep = 2
    sigma = minSigma/sigmastep
    while sigma < maxSigma:
        sigma = int(sigma*sigmastep)
        
        #make gaussian blur
        gblur = ndimage.gaussian_filter(img,0.4*sigma)
        sfilter = ndimage.sobel(gblur)
        
        #store data
        meta.append(keyname %sigma)
        features.append(sfilter)
        
    #stack features
    features = np.dstack([f for f in features])
         
    return meta, features

def Watershed_distance(img, threshmethod = filters.threshold_yen):

    #thresholding of image
    threshold = threshmethod(img)
    cells = img > threshold
    
    #mark distances
    distance = ndi.distance_transform_edt(cells)
    distance = np.expand_dims(distance, axis = -1)
    
    return ['Watershed_distance'], distance


    
def Meijering_filter(img,minSigma = 1,maxSigma = 16):
    scount = 0
    
    #no gaussian first
    keyname = 'Meijering_filter_%01d.0'  
    sfilter = filters.meijering(img)
    meta = [keyname %scount]
    features = [sfilter]
    scount = scount+1
     
    sigmastep = 2
    sigma = minSigma/sigmastep
    while sigma < maxSigma:
        sigma = int(sigma*sigmastep)
        
        #make gaussian blur
        gblur = ndimage.gaussian_filter(img,0.4*sigma)
        sfilter = filters.meijering(gblur)
        
        meta.append(keyname %sigma)
        features.append(sfilter)
        
    #stack features
    features = np.dstack([f for f in features])
         
    return meta, features

def Hessian(img,minSigma = 1,maxSigma = 16):
    scount = 0
    
    #no gaussian first
    keyname = 'Hessian_%01d.0'  
    sfilter = filters.meijering(img)
    meta = [keyname %scount]
    features = [sfilter]
    scount = scount+1
     
    sigmastep = 2
    sigma = minSigma/sigmastep
    while sigma < maxSigma:
        sigma = int(sigma*sigmastep)
        
        #make gaussian blur
        gblur = ndimage.gaussian_filter(img,0.4*sigma)
        sfilter = filters.hessian(gblur)
        
        meta.append(keyname %sigma)
        features.append(sfilter)
        
    #stack features
    features = np.dstack([f for f in features])
         
    return meta, features


def Sklearn_basic(img):
    basic = feature.multiscale_basic_features(img)
    return ['Sklearn_basic']*basic.shape[-1], basic




def Mean(img,minsigma=1,maxsigma=16):
    return Basic_filter(img,minsigma,maxsigma,np.mean,'Mean')

def Variance(img,minsigma=1,maxsigma=16):
    return Basic_filter(img,minsigma,maxsigma,np.var,'Variance')

def Median(img,minsigma=1,maxsigma=16):
    return Basic_filter(img,minsigma,maxsigma,np.median,'Median')

def Maximum(img,minsigma=1,maxsigma=16):
    return Basic_filter(img,minsigma,maxsigma,np.max,'Maximum')

def Minimum(img,minsigma=1,maxsigma=16):
    return Basic_filter(img, minsigma, maxsigma, np.min, 'Minimum')

##########################################

#Functions for handling training data and classifiers

def get_features(selectFeatures,img,minSigma = 1, maxSigma = 16, patchSize = 19, membraneSize = 1):
#run through feature functions and return list of features and 3d image (axis -1 is features)

    #dictionary to store results
    meta = ['original']
    features = np.expand_dims(img,axis = -1)
    
    i = 0
    for feat in selectFeatures:
        i+=1
        if callable(feat):
            m, f = feat(img)
        else:
            print('feature ' + str(i) + ' in featureselect is not callable and will be ignored')
            continue
        
        features = np.concatenate((features,f),axis = -1)
        meta = meta + m
        
    return(meta, features)


def import_training_data(imgdir,maskdir,ext = '.tif'):
#import training data -> expects directories for images and each labeled masks of the same name


    #get list of all files in imagedir
    if os.path.isdir(imgdir):
        filenames = glob.glob(imgdir + '/*' + ext)
    elif os.path.isfile(imgdir):
        filenames = [imgdir]
    else:
        print(imgdir + ' could not be found')
        return [],[]
    
    if not isinstance(maskdir,list):
        print('labels must be provided in a list')
        return [],[]
    
    #lists to store data
    IMG = []
    LABELS = []
    
    # print('importing images')
    
    #loop through files and import data
    for filename in filenames:

        # load images
        img = imread(filename)

        # Build an array of labels for training the segmentation.
        # Import mask image as labels
        training_labels = np.zeros(img.shape)

        i = 1
        for mdir in maskdir:
            try:
                mask = imread(filename.replace(imgdir,mdir))
            except:
                print('Could not find mask ' + filename.replace(imgdir,mdir))
                break
            if mask.shape != img.shape:
                print('Error: mask and image do not match shape',img.shape,mask.shape)
                break
            mask = (mask/np.max(mask))*i
            training_labels = np.add(training_labels, mask)
            i = i+1

        #only append when there are labels
        if i > 1:
            IMG.append(img)
            LABELS.append(training_labels)
        
    return IMG, LABELS

def pad_images(images):
#helper function for when images are different sizes 
    
    #get the largest dimensions
    xcheck = [i.shape[0] for i in images]
    ycheck = [i.shape[1] for i in images]
    newx = np.max(xcheck)
    newy = np.max(ycheck)
    
    #list to store new images
    NEW = []
    for img in images:
        newimg = np.zeros((newx,newy))
        #write image to top left of new image
        newimg[0:img.shape[0],0:img.shape[1]] = img
        NEW.append(newimg)
        
    return NEW


def get_training_data(IMG,LABELS,featureselect,loaddatafile = None, savedatafile = None,returnmeta = False):
#gets training data

    #catch if you only pass in one image or label or feature function and put them in lists
    if not isinstance(IMG,list):
        if isinstance(IMG,np.ndarray):
            IMG = [IMG]
        else:
            raise Exception('images in incorrect format; expected ndarray')

    if not isinstance(LABELS,list):
        if isinstance(LABELS,np.ndarray):
            LABELS = [LABELS]
        else:
            raise Exception('labels in incorrect format; expected ndarray')

    if len(IMG) != len(LABELS):
        raise Exception('mismatch between number of labeled images and input images')
        pass
    
    if not isinstance(featureselect,list):
        if callable(featureselect):
            featureselect = [featureselect]
        else:
            raise Exception('Featureselect is not callable')

    else:
        if not any([callable(feat) for feat in featureselect]):
            raise Exception('featureselect is not callable')
            
        
    #check if image sizes are equal
    sizecheck = [f.shape[0]*f.shape[1] for f in IMG]
    if not all(elem == sizecheck[0] for elem in sizecheck):
        IMG = pad_images(IMG)
        LABELS = pad_images(LABELS)
    
    # get features from list of images
    featuredata = [get_features(featureselect, simg) for simg in IMG]
    meta, FEATURES = list(zip(*featuredata)) 
        
    if loaddatafile is not None:
        loadmeta, loadFEATURES,loadLABELS,loadfeatureselect = load_training_features(loaddatafile)
        if loadfeatureselect == featureselect:
            meta = meta + loadmeta
            FEATURES = FEATURES + loadFEATURES
            LABELS = LABELS + loadLABELS
        else:
            print('loaded data file has different feature selection and is not included')
       
    
    # flatten trainingfeatures for classifier
    trainingfeatures = np.concatenate(FEATURES)
    trainingfeatures = trainingfeatures.reshape(-1, trainingfeatures.shape[-1])
    
    # flatten traininglabels for classifier
    traininglabels = np.concatenate(LABELS)
    traininglabels = traininglabels.flatten()
    
    #isolate labeled pixels and remove unlabeled pixels
    trainingfeatures = trainingfeatures[traininglabels > 0,:]
    traininglabels = traininglabels[traininglabels > 0]
    
    if savedatafile is not None:
        pickle.dump([meta,FEATURES,LABELS,featureselect],open(savedatafile,'wb'))
    
    if returnmeta:
        return traininglabels, trainingfeatures, meta
    else:
        return traininglabels, trainingfeatures

def load_training_features(loaddatafile):
    
    meta, FEATURES, LABELS, featureselect = pickle.load(open(loaddatafile, 'rb'))
    
    return meta, FEATURES, LABELS, featureselect

def load_training_data(loaddatafile, returnmeta = False):
    meta, FEATURES, LABELS, featureselect = load_training_data(loaddatafile)
    
    # flatten trainingfeatures for classifier
    trainingfeatures = np.concatenate(FEATURES)
    trainingfeatures = trainingfeatures.reshape(-1, trainingfeatures.shape[-1])
    
    # flatten traininglabels for classifier
    traininglabels = np.concatenate(LABELS)
    traininglabels = traininglabels.flatten()
    
    #isolate labeled pixels and remove unlabeled pixels
    trainingfeatures = trainingfeatures[traininglabels > 0,:]
    traininglabels = traininglabels[traininglabels > 0]
    
    if returnmeta:
        return traininglabels, trainingfeatures, featureselect, meta
    else:
        return traininglabels, trainingfeatures, featureselect

def load_classifier(clffile):
    
    clf, featureselect = pickle.load(open(clffile, 'rb'))
    
    return clf, featureselect

def train_classifier(traininglabels,trainingfeatures,featureselect, saveclftofile = None, clf = None):
#train classifier -> best practice at least two labels (e.g. object, not object)
#if clf is None default classifier is RandomForest
#IMG and LABELS are lists of images and masks respectively
#features select is list of feature functions herein
    
    # train classifier
    if clf is None:
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                     max_depth=10, max_samples=0.05)
    clf = clf.fit(trainingfeatures, traininglabels)
    
    #saves clf to file if given filename (any string - saves relative to working directory)
    if not saveclftofile is None:
        pickle.dump([clf,featureselect], open(saveclftofile, 'wb'))
    
    return clf

def classify_image_probability(img,clf,featureselect):
#classify image with classifier and return probablity of label == 1 as image

    # get features of image and flatten
    meta, features = get_features(featureselect, img)
    features = features.reshape(-1, features.shape[-1])

    # predict probability
    result = clf.predict_proba(features)
    
    # convert to image
    result = np.reshape(result, (img.shape[0], img.shape[1], result.shape[-1]))

    return result

def classify_image(img,clf,featureselect):
#classify image with classifier and return result
    
    # get features of image and flatten
    meta, features = get_features(featureselect, img)
    features = features.reshape(-1, features.shape[-1])

    # predict probability
    result = clf.predict(features)
    
    # convert to image
    result = np.reshape(result, (img.shape[0], img.shape[1]))
    
    return result

def classify_image_label(img,clf,featureselect,selectlabel = 1):
#classify image with classifier and return probablity of label == selectlabel as image

    # get features of image and flatten
    meta, features = get_features(featureselect, img)
    features = features.reshape(-1, features.shape[-1])

    # predict probability
    result = clf.predict(features)
    # convert to image
    resultimg = np.reshape(result, (img.shape[0], img.shape[1]))
    # get specifically label == selectlabel
    label = np.zeros(img.shape)
    label[np.where(resultimg == selectlabel)] = 1
    
    return label

def classify_image_label_probability(img,clf,featureselect,selectlabel = 1):
#classify image with classifier and return probablity of label == selectlabel as image

    # get features of image and flatten
    meta, features = get_features(featureselect, img)
    features = features.reshape(-1, features.shape[-1])

    # predict probability
    result = clf.predict_proba(features)
    # convert to image
    result = np.reshape(result, (img.shape[0], img.shape[1], result.shape[-1]))
    # get specifically label == 1
    label = result[:, :, selectlabel]
    
    return label


def threshold_mask(img,threshmethod = filters.threshold_minimum):
#threshold an image using threshmethod -> sklearn_filters.threshold_*
     thresh = threshmethod(img)
     ##make binary mask
     mask = img > thresh
     mask = mask*255/np.max(mask)
     
     return mask





