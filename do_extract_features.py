"""Extracts three types of features from standardized square images:
        (1) Wing texture: Gabor transformation features
        (2) Wing color: Regional intensities of pixels in chromatic coords
        (3) Wing shape: coefficients from morphometric analysis
    Features are saved to a csv file.
"""

### Setup ======================================================================
print ('\nSetting up ...')

import os#, sys
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
import autoID

# Read config file
cfg = autoID.utils.read_config('config.yaml')

# Initiate log file
#try: log_extract = pd.read_csv(log_extract_path,header=0,index_col=0) # load log
#except IOError: log_extract = pd.DataFrame(columns=['Error']) # create new log

# Import metadata
md = pd.read_csv(cfg['metadata_path'],header=0,index_col=None,encoding='utf-8')

# Import previously-extracted features
try: fd = pd.read_csv(cfg['features_path'],header=0,index_col=0)
except IOError:
    columns = np.arange(1,663+1).astype(str)
    fd = pd.DataFrame(columns=columns)


### Look for images to extract from ============================================
print ('Finding images from which to extract features ...')

# Get square filenames
square_files = [f for f in os.listdir(cfg['squares_path']) if f.endswith('.tif')]

# Which images need extracting?
image_cue = list(set(square_files) - set(fd.index.values))
print (' - Found {} images to process'.format(len(image_cue)))

### Extract features from images ===============================================
print ('Extracting ...')
t0 = datetime.now()

# Extract features from them & add them to extracted_features
yep = 0; nope = 0 #succes/fail counters
for i,fn in enumerate(image_cue):
    # Print status update for every 10th image
    if (i+1)%10 == 0:
        print (' - Preprocessing image {} of {}. {} successes, {} failures so far ...'\
               .format(i+1,len(image_cue),yep,nope))

    # TODO: make this a function

    # Load image
    try:
        square = cv2.imread(os.path.join(cfg['squares_path'],fn))[:,:,::-1]
    except:
        #appendRowIf(log_extract,fn,'Error loading image.')
        nope+=1; continue

    # Extract features
    try:
        morphCoeffs = autoID.extraction.morphometric_sample(square) # 0.506225 s
    except RuntimeError:
        #appendRowIf(log_extract,fn,"Error extracting 'morph' features: {}".format(os.environ['error_message']))
        nope+=1; continue
    except:
        #appendRowIf(log_extract,fn,"Error extracting 'morph' features.")
        nope+=1; continue
    try:
        chromCoeffs = autoID.extraction.chrom_sample(square) # 0.17824 s
    except:
        #appendRowIf(log_extract,fn,"Error extracting 'chrom' features.")
        nope+=1; continue
    try:
        gaborCoeffs = map(lambda x: autoID.extraction.gabor_sample(square,x,3),[8,4,2,1])
        gaborCoeffs = np.concatenate(gaborCoeffs) # 1.725601 s
    except:
        #appendRowIf(log_extract,fn,"Error extracting 'gabor' features.")
        nope+=1; continue

    # Concatenate features
    features = []
    features.extend(list(morphCoeffs))
    features.extend(list(chromCoeffs))
    features.extend(list(gaborCoeffs))

    # Append features to dataframe
    fd.loc[fn] = features

    yep+=1

time = datetime.now() - t0
print (' - Extraction done: {} successes, {} failures. Took {}'.format(yep,nope,time))


### Save out features & clean up ===============================================
print ('Saving features ...')
fd.to_csv(cfg['features_path'],header=True,index=True,index_label='img_filename')
print ('Done. Clean metadata save to: {}'.format(cfg['features_path']))

# Save log file
print ('Finishing up ...')
#log_extract.to_csv(log_extract_path,header=True,index=True,index_label='Filename')
print ('Done.')
