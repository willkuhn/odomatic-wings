"""Use a pre-fitted model to classify a sample from image, standardized
    square image or pre-extracted features.

    Example usage:
    > python do_use_classifier.py -m rf_24cl_82ac.pkl -i /media/sf_All_scan_images/WRK-WS-0613_s.tif

    > python do_use_classifier.py -m rf_24cl_82ac.pkl -i "D:/Desktop/All scan images/WRK-WS-00436_Pantala_flavescens_M_s.tif"
"""

### Setup ======================================================================

import os#, sys
from datetime import datetime
import pandas as pd
import numpy as np
import autoID
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline#,Pipeline
import cv2
#import matplotlib.pyplot as plt
import pickle #import cPickle as pickle
#from sklearn.externals import joblib
import argparse


# Parse arg: whether to save model
ap = argparse.ArgumentParser()
ap.add_argument('-m','--model', action='store', required=True, type=str,
    help="Name of model pickle to use (stored in /models/)")
ap.add_argument('-i','--image', action='store', required=False, default=None,
    type=str, help='Filepath to image on which to predict')
ap.add_argument('-q','--square', action='store', required=False, default=None,
    type=str, help='Filepath to square image on which to predict')
ap.add_argument('-f','--features', action='store', required=False, default=None,
    type=str, help='Filename of image from which features have been pre-extracted')

args = vars(ap.parse_args())
model_fn = args['model']
image_fp = args['image'] # defaults to None
square_fp = args['square'] # defaults to None
features_key = args['features'] # defaults to None

print ('\nSetting up ...')

# Read config file
cfg = autoID.utils.read_config('config.yaml')

"""
# DEBUG
model_fn = 'rf_24cl_82ac.pkl'
image_fp = os.path.join(image_path,'Polythore_spaeteri_M-00760_s.tif')
"""


### Get model ==================================================================
print ('Getting model ...')

model_fp = os.path.join(cfg['model_path'],model_fn) #os.getcwd()
try:
    with open(model_fp,'rb') as f:
        modelPacket = pickle.load(f)
    #modelPacket = joblib.load(f)
except IOError: raise IOError('No model found at {!r}'.format(model_fp))

# Import the appropriate modules from `skimage` to run model
clfMethod = modelPacket['params']['clfMethod']
clf = autoID.classifier.setup_classifier(cfg['clfMethod']) # set up classifier

# Create prediction pipeline for new features
ss = modelPacket['scaler']
pca = modelPacket['pca']
clf = modelPacket['model']
pipeline = make_pipeline(ss,pca,clf)


### Process input ==============================================================
print ('Processing input ...')

if image_fp: # load image and standardize it
    t0 = datetime.now()

    # Load image
    try:
        img = cv2.imread(image_fp)[:,:,::-1] # Load & switch from BGR to RGB
    except:
        raise IOError('Unable to import image.')

    # Convert it to square
    try:
        square = autoID.preprocessing.make_square(img)
    except:
        raise RuntimeError('Error preprocessing image to square.')

    # Extract features
    try:
        morphCoeffs = autoID.extraction.morphometric_sample(square)
        chromCoeffs = autoID.extraction.chrom_sample(square)
        gaborCoeffs = map(lambda x: autoID.extraction.gabor_sample(square,x,3),[8,4,2,1])
        gaborCoeffs = np.concatenate(gaborCoeffs)
    except:
        raise RuntimeError('Error extracting features from square.')

    # Concatenate features
    features = []
    features.extend(list(morphCoeffs))
    features.extend(list(chromCoeffs))
    features.extend(list(gaborCoeffs))

    time = datetime.now() - t0
    print (' - Preprocessing from image took {} (h:m:s)'.format(time))

elif square_fp:
    t0 = datetime.now()

    # Load square image
    try:
        square = cv2.imread(square_fp)[:,:,::-1] # switch from BRG to RGB
    except:
        raise IOError('Unable to import square image.')

    # Extract features
    try:
        morphCoeffs = autoID.extraction.morphometric_sample(square)
        chromCoeffs = autoID.extraction.chrom_sample(square)
        gaborCoeffs = map(lambda x: autoID.extraction.gabor_sample(square,x,3),[8,4,2,1])
        gaborCoeffs = np.concatenate(gaborCoeffs)
    except:
        raise RuntimeError('Error extracting features from square.')

    # Concatenate features
    features = []
    features.extend(list(morphCoeffs))
    features.extend(list(chromCoeffs))
    features.extend(list(gaborCoeffs))

    time = datetime.now() - t0
    print (' - Preprocessing from standardized image took {} (h:m:s)'.format(time))

elif features_key:
    # Import features
    fd = pd.read_csv(cfg['features_path'],header=0,index_col=0)
    try:
        features = fd.loc[features_key]
    except KeyError:
        raise RuntimeError('Unable to find {} in pre-extracted features dataset'.format(features_key))

    print (' - Features loaded for {!r}'.format(features_key))



### Make prediction & print results ===========================================

if hasattr(pipeline,'predict_proba'):
    t0 = datetime.now()
    probs = pipeline.predict_proba([features])[0]
    time = datetime.now() - t0

    classes = pipeline.classes_
    order = np.argsort(probs)[::-1] # decending sort order

    print ('Predicting done in {} (h:m:s).'.format(time))
    print ('\nPredicted class by probability:\n'+'='*30)
    for c,p in zip(classes[order],probs[order]):
        print ('{:<25}{:>8.2f}%'.format(c,p*100))

else:
    t0 = datetime.now()
    predicted = pipeline.predict([features])
    time = datetime.now() - t0
    print ('Predicting done in {} (h:m:s).'.format(time))
    print (' - Probabilities unavailable with this classifier.')
    print (' - Predicted class: {!r}'.format(predicted[0]))
