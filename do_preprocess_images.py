"""Preprocesses images (scans of 2 odonate wings) for identification by
    converting them into into standardized, 512-px squares. Unsuccessful
    conversions are logged to a table."""

### Set up =====================================================================
print ('\nSetting up ...')

# Load packages
import os
#os.chdir('D:\\Dropbox\\dev\\github-repos\\odomatic-wings')
#import pandas as pd
import cv2
from datetime import datetime
#import skimage
#import scipy.ndimage as nd
import autoID

#def appendRowIf(df,filename,error): # for error logging
#    # adds {filename:error} to df if filename is not in df's index
#    if not filename in df.index.values: df.loc[filename] = error

# Read config file
cfg = autoID.utils.read_config('config.yaml')

# Initiate log file
#try: log_preproc = pd.read_csv(log_preproc_path,header=0,index_col=0) # load log
#except IOError: log_preproc = pd.DataFrame(columns=['Error']) # create new log


### Find images that haven't been preprocessed =================================
print ('Finding images to preprocess ...')

# Get list of tif image filenames from image_path
image_files = [f for f in os.listdir(cfg['image_path']) if f.endswith('.tif')]
# Get square filenames or create dir in squares_path
if os.path.isdir(cfg['squares_path']):
    square_files = [f for f in os.listdir(cfg['squares_path']) if f.endswith('.tif')]
else:
    os.mkdir(cfg['squares_path'])
    square_files = []

# Which images need processing?
image_cue = list(set(cfg['image_files']) - set(cfg['square_files']))

print (' - Found {} unprocessed images'.format(len(image_cue)))



### Preprocess images ====================================================
print ('Processing images ...')
t0 = datetime.now()

# Loop through cue, converting images into squares & saving them out
yep = 0; nope = 0 #succes/fail counters
for i,fn in enumerate(image_cue):
    # Print status update for every 10th image
    if (i+1)%10 == 0:
        print (' - Preprocessing image {} of {}'.format(i+1,len(image_cue)))

    # TODO: make this a function

    # Load image
    image_fp = os.path.join(cfg['image_path'],fn)
    try:
        img = cv2.imread(image_fp)
    except:
        #appendRowIf(log_preproc,fn,'Error loading image.')
        nope+=1; continue
    img = img[:,:,::-1] # Switch from BGR to RGB

    # Convert it to square
    try:
        square = autoID.preprocessing.make_square(img)
    except RuntimeError: #Log filename and error (from environmental variable `error_message`)
        #appendRowIf(log_preproc,fn,os.environ['error_message'])
        nope+=1; continue
    except:
        #appendRowIf(log_preproc,fn,'Preprocessing error: unspecified.')
        nope+=1; continue

    # Save square image
    square = square[:,:,::-1] #Switch back from RGB to BGR
    cv2.imwrite(os.path.join(cfg['squares_path'],fn),square)

    yep+=1

time = datetime.now() - t0
print (' - Preprocessing done: {} successes, {} failures. Took {}'.format(yep,nope,time))

### Finish up ==================================================================
print ('Finishing up ...')
# Save log file
#log_preproc.to_csv(log_preproc_path,header=True,index=True,index_label='Filename')
