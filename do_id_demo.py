"""Demo of identification from wing scans: ID from random images in `image_path`
    with simple UI.

    Example usage:
    > python do_id_demo.py -m rf_24cl_82ac.pkl

"""

### Setup ======================================================================

import os, random#, sys
from datetime import datetime
import pandas as pd
import numpy as np
import autoID
from sklearn.pipeline import make_pipeline#,Pipeline
import cv2
import pickle
import argparse

def handler(event, x, y, flags, param): #mouse callback function
    #based on Adrian Rosebrock's post: 'capturing mouse click events with python
    #and opencv'
    if event == cv2.EVENT_LBUTTONDOWN:
    	pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass


# Parse arg: whether to save model
ap = argparse.ArgumentParser()
ap.add_argument('-m','--model', action='store', required=True, type=str,
    help="Name of model pickle to use (stored in /models/)")

args = vars(ap.parse_args())
model_fn = args['model']

print ('\nSetting up ...')

# Load config file, if not already done
cfg = autoID.utils.read_config('config.yaml')

# Import metadata, add Name & Label cols & make dict to look up name & labels
md = pd.read_csv(cfg['metadata_path'],header=0,index_col=None)
md['Name'] = [' '.join(i) for i in md[cfg['groupClassBy']].values]
gb = md.groupby('Name').count()['Species'] # Groupby name & tally each name
rare_classes = gb[gb<cfg['minClassSize']].index # classes with insufficient indivs
md['Label'] = md['Name'] # copy 'Name' column
md.loc[md['Label'].isin(rare_classes),'Label'] = 'rare_unknown' # relabel rares
fn2name = dict(zip(md['img_filename'],md['Name'])) #translates filename to name
fn2label = dict(zip(md['img_filename'],md['Label'])) #translates filename to label

# Get model
model_fp = os.path.join(cfg['model_path'],model_fn) #os.getcwd()
try:
    with open(model_fp,'rb') as f:
        modelPacket = pickle.load(f)
    #modelPacket = joblib.load(f)
except IOError: raise IOError('No model found at {!r}'.format(model_fp))
# Import the appropriate modules from `skimage` to run model
clfMethod = modelPacket['params']['clfMethod']
clf = autoID.classifier.setup_classifier(clfMethod) # set up classifier

# Create prediction pipeline for new features
ss = modelPacket['scaler']
pca = modelPacket['pca']
clf = modelPacket['model']
pipeline = make_pipeline(ss,pca,clf)

# Set up image cue
# Get image filenames from squares path from which to select random images
image_cue = list(set(os.listdir(cfg['squares_path'])) & set(md['img_filename']))
# Shuffle the cue
random.shuffle(image_cue)


### Initiate UI ================================================================

print ('Starting UI...')

running = True #whether while statement is running
start = True #if True, runs next image, which waits for keypress at end
key = None #default keypress value

cv2.namedWindow('Original image')
cv2.setMouseCallback('image', handler)

#cv2.startWindowThread()
while running:

    while start:
        # Get filepath for next image in cue
        cv2.destroyAllWindows()
        image_fn = image_cue.pop() # use this to look up true name later
        image_fp = os.path.join(cfg['image_path'],image_fn)
        t0 = datetime.now()

        print ('Processing image {!r}...'.format(image_fn))
        ## Preprocess image
        # Load it
        try:
            img = cv2.imread(image_fp)[:,:,::-1] # Load & switch from BGR to RGB
            cv2.imshow('Original image',img[:,:,::-1])
        except:
            raise IOError('Unable to import image.')
        # Convert it to square
        try:
            square = autoID.preprocessing.make_square(img)
            cv2.imshow('Standardized image',square[:,:,::-1])
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
        print (' - Preprocessing took {} (h:m:s)'.format(time))

        # Make prediction & print results
        print ('Predicting ...')
        if hasattr(pipeline,'predict_proba'):
            t0 = datetime.now()
            probs = pipeline.predict_proba([features])[0]
            time = datetime.now() - t0

            classes = pipeline.classes_
            order = np.argsort(probs)[::-1] # decending sort order

            print (' - Prediction finished in {} (h:m:s).'.format(time))
            print ('\nTrue class: {!r}'.format(fn2name[image_fn]))
            print ('True label: {!r}'.format(fn2label[image_fn]))
            print ('\nPredicted class by probability:\n'+'='*30)
            print ('\n'.join(['{:<25}{:>8.2f}%'.format(c,p*100) for c,p in zip(classes[order],probs[order])]))
            print ('\n')

        else:
            t0 = datetime.now()
            predicted = pipeline.predict([features])
            time = datetime.now() - t0
            print ('Prediction finished in {} (h:m:s).'.format(time))
            print ('True class: {!r}'.format(fn2name[image_fn]))
            print ('True label: {!r}'.format(fn2label[image_fn]))
            print ('\nPredicted class: {!r}'.format(predicted[0]))
            print ('\n')

        print ("`n` to go to next image\n`q` to quit") # instructions

        start = False
        key = cv2.waitKey(0) #& 0xFF


	# if the 'n' key is pressed, go to next image in cue:
    if key == ord('n'):
        print ('\nNext image ...\n')
        start = True #initiate white-start loop

	# if the 'q' key is pressed, break from while-running loop:
    elif key == ord('q'):
        print ('- Quitting')
        running = False #exit while-running loop
