"""
UI to gather positive & negative training examples for training
the masking model

Note: Full scratch script with extra analyses located at:
WING SCANING PROJECT/dev/scratch-masking.py

@author: Will
"""
### Imports & functions ========================================================
import os
import cv2
import pandas as pd
import autoID
from autoID.preprocessing import transparency_transform

# class to store points during data collection
class CoordinateStore:
    """Source for CoordinateStore & point selector window: Mr Vinagi @ http://stackoverflow.com/a/39048325/7190721

    """
    def __init__(self,fn,pos_or_neg):
        columns = ['filename','type','x','y','Cr','Cb','S','MC6','MC12','E10']

        self.data =         pd.DataFrame(data=[],columns=columns)
        self.fn =           fn
        self.pos_or_neg =   pos_or_neg
        if pos_or_neg == 'pos':
            self.color = (0,255,0) # green
        elif pos_or_neg == 'neg':
            self.color = (0,0,255) #red (in a BGR image)

    def select_point(self,event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:#cv2.EVENT_LBUTTONDBLCLK:
                # draw a dot at point x,y
                cv2.circle(clone,(x,y),3,self.color,-1)
                # grab feature vector for point x,y
                cr,cb,s,m6,m12,e10 = img_trans[y,x]
                # add row to instance's dataframe
                data = { 'filename':    self.fn,
                         'type':        self.pos_or_neg,
                         'x':           x,
                         'y':           y,
                         'Cr':          cr,
                         'Cb':          cb,
                         'S':           s,
                         'MC6':         m6,
                         'MC12':        m12,
                         'E10':         e10}
                self.data = self.data.append(data,ignore_index=True)

    def reset(self): # reset instances dataframe
        columns = ['filename','type','x','y','Cr','Cb','S','MC6','MC12','E10']
        self.data =         pd.DataFrame(data=[],columns=columns)


### Setup ======================================================================

# Read config file
cfg = autoID.utils.read_config('config.yaml')

im_path = cfg['images_for_masking_model_path']
filenames = [i for i in os.listdir(im_path) if i.endswith('tif')]
filenames = [i for i in filenames if not i.startswith('None')]


### Collect data ===============================================================

# Set up training dataframe (only do once, otherwise import from CSV)
#columns = ['filename','type','x','y','Cr','Cb','S','MC6','MC12','E10']
#train_df = pd.DataFrame(data=[],columns=columns)

# Import train_df from CSV
train_df = pd.read_csv(cfg['masking_data_path'],header=0,index_col=None)

# Set up image cue
image_cue = list(set(filenames) - set(train_df['filename']))

i = 0 # counter
go = True

# GUI LOOP
while(go):
    if i<len(image_cue):
        fn = image_cue[i]
    else:
        print('No more images to annotate.')
        go = False

    img = cv2.imread(os.path.join(im_path,fn)) # get image as BGR
    clone = img.copy() # make copy of image to draw over
    #plt.imshow(img)
    # transform RGB image
    img_trans = transparency_transform(img[:,:,::-1])

    # GET POSITIVE TRAINING EXAMPLES:

    #instantiate class for positive training examples
    pos = CoordinateStore(fn,'pos')

    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',pos.select_point)

    # Pos loop
    while(1):
        cv2.imshow('image',clone)
        k = cv2.waitKey(20) & 0xFF
        if k == 32: # SPACE: save examples then exit pos loop
            train_df = train_df.append(pos.data,ignore_index=True)
            break
        elif k == ord('r'): # 'r': reset points
            pos.reset() # reset class instance
            clone = img.copy() # reset display image
        elif k == 27: # ESC: exit outer loop without saving
            go = False
            break
    cv2.destroyAllWindows()


    # GET NEGATIVE TRAINING EXAMPLES:
    img = clone.copy() # update image with current points in case need to reset neg points

    #instantiate class for positive training examples
    neg = CoordinateStore(fn,'neg')

    # Create a black image, a window and bind the function to window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',neg.select_point)

    # Neg loop
    while(1):
        cv2.imshow('image',clone)
        k = cv2.waitKey(20) & 0xFF
        if k == 32: # SPACE: save examples then exit neg loop
            train_df = train_df.append(neg.data,ignore_index=True)
            break
        elif k == ord('r'): # 'r': reset points
            neg.reset() # reset class instance
            clone = img.copy() # reset display image
        elif k == 27: # ESC: exit outer loop without saving
            go = False
            break
    cv2.destroyAllWindows()

    i += 1 # advance the counter

print('\nPoints so far: {} pos + {} neg = {} total'.format(train_df[train_df['type']=='pos'].shape[0],train_df[train_df['type']=='neg'].shape[0],len(train_df)))

# Overwrite CSV with values from updated train_df
train_df.to_csv(cfg['masking_data_path'],header=True,index=False)
