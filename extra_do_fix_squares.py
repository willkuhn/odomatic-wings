"""Some wings in my dataset were scanned incorrectly (hindwing over forewing,
    'switchies') and others were left wings, not right wings (i.e. 'lefties').
    This script fixes those issues in my dataset.
    """

### Set up =====================================================================
print ('\nSetting up ...')

# Load packages
import os,sys
#os.chdir('D:\\Dropbox\\dev\\github-repos\\odomatic-wings')
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

# Load config file, if not already done
try: config_loaded
except NameError:
    f = 'config_linux.txt' if 'linux' in sys.platform else 'config.txt'
    execfile(os.path.join(os.getcwd(),f))

# Get metadata
md = pd.read_csv(metadata_path,index_col=None,header=0,encoding='utf-8')

### Reflect left wings to match the right ones:
print ('Flipping lefties ...')
t0 = datetime.now()

# Find lefties
lefties = md.loc[md['Condition'].str.contains('left',na=False,case=False),'img_filename']
lefties = lefties.values.astype(str)
lefties = [f for f in lefties if f in os.listdir(squares_path)]
#lefties = ['WRK_WS_01826_Epitheca_cynosura_F_s.tif']
print (' - Found {} images to flip'.format(len(lefties)))

# Flip them
for fn in lefties:
    img = cv2.imread(os.path.join(squares_path,fn))
    img = np.fliplr(img) # Reflect image left-to-right
    cv2.imwrite(os.path.join(squares_path,fn), img)
    print (' - Flipped {}'.format(fn))

#for n in ['895','1584','2509','2486']:
time = datetime.now()-t0
print (' - Done in {}'.format(time))

### Switch fore- and hingwings for some specimens:
print ('Switching switchies ...')
t0 = datetime.now()

# Find 'switchies'
switchies = md.loc[md['Condition'].str.contains('switch',na=False,case=False),'img_filename']
switchies = switchies.values.astype(str)
switchies = [f for f in switchies if f in os.listdir(squares_path)]
#switchies = ['WRK_WS_1696_Libellulidae_sp__s.tif']
print (' - Found {} images to switch'.format(len(switchies)))

# Switch them
for fn in switchies:
    img = cv2.imread(os.path.join(squares_path,fn))
    length=len(img)
    top = img[:length//2]
    bottom = img[length//2:]
    img = np.concatenate((bottom,top))#Switch top & bottom halves
    cv2.imwrite(os.path.join(squares_path,fn), img)
    print (' - Switched {}'.format(fn))

time = datetime.now()-t0
print (' - Done in {}'.format(time))
