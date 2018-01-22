"""
Re-extract features from pos & neg points (as training data for masking model)

Transforms images in `masking_data_path` file and re-extracts feature
vectors from the coordinates specified in that file, then overwrites
the file with the new features.

*Only use this if the transformation scheme is changed for the feature
extraction part of the wing-masking algorithm.

@author: Will
"""
### Imports & functions ========================================================
import os
import cv2
import pandas as pd
import autoID
from autoID.preprocessing import transparency_transform

# Read config file
cfg = autoID.utils.read_config('config.yaml')

im_path = cfg['images_for_masking_model_path']
filenames = [i for i in os.listdir(im_path) if i.endswith('tif')]
filenames = [i for i in filenames if not i.startswith('None')]

# Set up dataframe for new feature vectors (only do once, otherwise import from CSV)
columns = ['filename','type','x','y','Cr','Cb','S','MC6','MC12','E10']
train_df2 = pd.DataFrame(data=[],columns=columns)

# Import train_df from CSV
train_df = pd.read_csv(cfg['masking_data_path'],header=0,index_col=None)

# Set up image cue
image_cue = set(train_df['filename'].values)

# Get re-retreive values from image transformation into new dataframe: train_df2
for fn in image_cue:
    print(fn)
    img = cv2.imread(os.path.join(im_path,fn)) # get image as BGR
    img_trans = transparency_transform(img[:,:,::-1])

    df = train_df.loc[train_df['filename']==fn]
    for filename,pos_or_neg,x,y in df[['filename','type','x','y']].values:
                cr,cb,s,m6,m12,e10 = img_trans[int(y),int(x)]
                # add row to instance's dataframe
                data = { 'filename':    filename,
                         'type':        pos_or_neg,
                         'x':           x,
                         'y':           y,
                         'Cr':          cr,
                         'Cb':          cb,
                         'S':           s,
                         'MC6':         m6,
                         'MC12':        m12,
                         'E10':         e10}
                train_df2 = train_df2.append(data,ignore_index=True)

# Overwrite CSV with values from updated train_df
train_df2.to_csv(cfg['masking_data_path'],header=True,index=False)
