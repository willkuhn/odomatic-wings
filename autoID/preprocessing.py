# -*- coding: utf-8 -*-
# Copyright (C) 2015 William R. Kuhn
"""
===============================================================================
IMAGE PREPROCESSING
===============================================================================
"""

import numpy as np
import cv2
import scipy.ndimage as nd
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, label
from sklearn.externals import joblib
from .utils import resize,pyramid,sliding_window

# Functions for pre-masking image transformation ===============================

def michelson_constrast(arr):
    """Calculates Michelson's contrast for an array of intensity values:
        MC = (Imax-Imin)/(Imax+Imin)

    MC is returned as int between 0-255. Handles zero divisions.
    """
    mx,mn = float(arr.max()),float(arr.min()) # max and min values
    if mx==0.: # catch zero division, assumes mn is not negative
        return 0
    else:
        return int(((mx-mn)/(mx+mn))*255)

def michelson_constrast_transform(image):
    """Calculates Michelson's contrast on a single-channel intensity image.

    Parameters
    ----------
    image : array-like
        single-channel intensity image cast as np.uint8

    Returns
    -------
    michelson : list of arrays
        Michelson's contrast of `image` at scales 6x6 and 12x12-px as uint8
        arrays

    Sources
    -------
    pyramids: http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    sliding windows: http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    # Get dimensions of original image
    h0,w0 = image.shape

    # Params for sliding window transformation
    winH,winW = (3,3) # window size; must be odd numbers
    stepSize = 1 # step size

    michelson = []

    # loop over the 1/2 & 1/4-sized images from image pyramid
    for i,resized in enumerate(pyramid(image, scale=2.,minSize=(100,100))):
        if i==0 or i>2:
            continue
        h,w = resized.shape[:2]
        out = np.zeros((h,w),dtype=np.uint8)

    	    # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            out[y,x] = michelson_constrast(window)

        michelson.append(out)

    # rescale transformed images back to size of `image`
    for i in range(len(michelson)):
        michelson[i] = cv2.resize(michelson[i],(w0,h0),interpolation=cv2.INTER_NEAREST)
    return michelson

def transparency_transform(image,bgr=False):
    """Returns a 6-channel transformation of an RGB image.

    Channels:
        (1) Cr (red-diff chroma) of YCrCb colorspace [1]
        (2) Cb (blue-diff chroma) of YCrCb colorspace [1]
        (3) S (saturation) of HSV colorspace [1]
        (4) MC6 : 6-px-window Michelson Contrast of Y (intens) of YCrCb [1,2]
        (5) MC12 : 12-px-window Michelson Contrast of Y [1,2]
        (6) E10 : Canny edge filtering + 10-px-radius blurring of Y

    Parameters
    ----------
    image : uint8 array, shape (h,w,3)
        Input image. Channels must be ordered RGB (not cv2's BGR)!
    bgr : bool, optional (default False)
        Whether to expect `image` channels to be in order BGR (as from
        cv2.imread()). Otherwise, channels assumed to be RGB

    Returns
    -------
    transformed_image : uint8 array, shape (h,w,6)
        Transformed image

    Sources:
    [1] Kompella, V.R., and Sturm, P. (2012). Collective-reward based approach
        for detection of semi-transparent objects in single images. Computer
        Vision and Image Understanding 116, 484–499.
    [2] https://en.wikipedia.org/wiki/Contrast_(vision)#Formula

    """
    if bgr: # convert image from BGR -> RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    yy,cr,cb = cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb))
    h,s,v = cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2HSV))
    mc6,mc12 = michelson_constrast_transform(yy)

    e10 = cv2.blur(cv2.Canny(yy,100,200),(21,21))
    return np.dstack([cr,cb,s,mc6,mc12,e10])


# Functions supporting the Standardizer class ==================================

def _check_image_mask_match(image,mask):
    return image.shape[:2]==mask.shape

def _load_model(filepath):
    return joblib.load(filepath)

def bounding_box(image,background='k'):
    """Get a bounding box for a mask or masked image. Image should only
    contain a single object. If image is color, its background color can
    be specified.

    Parameters
    ----------
    image : array, shape (h,w) or (h,w,3), dtype bool or uint8
        Input image. Should either be a bool mask or an image that has been
        masked, where the background is solid white or black.
    background : ('k'|'w'), optional (default 'k')
        Used if image is shape (h,w,3). Specifies background color of
        input image. 'k' for black or 'w' for white

    Returns
    -------
    output : tuple of ints
        Tuple of ints (min_row,min_col,max_row,max_col)

    Notes
    -----
    Background pixels in input image are presumed to be 0 or 0. and foreground
    pixels are >0. In this function, the first and last non-zero rows and
    columns are determined.
    """
    if image.ndim==3: # image is color
        g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # map non-background pixels
        m = np.zeros(g.shape,dtype=bool)
        if background is 'k':
            m[g>0] = True
        elif background is 'w':
            m[g<255] = True
        else:
            raise ValueError('`background` value not understood.')

    elif image.dtype in (bool,np.bool): # image is a boolean mask
        m = image

    else:
        raise ValueError('`image` invalid.')

    # collapse `m` row-wise and column-wise
    rows = np.any(m,axis=1)
    cols = np.any(m,axis=0)

    # get indices for rows and cols that contain something non-background
    occupied_row_inds = np.arange(len(rows))[rows]
    occupied_col_inds = np.arange(len(cols))[cols]

    rmin = max(0,occupied_row_inds[0]-1)
    cmin = max(0,occupied_col_inds[0]-1)
    rmax = occupied_row_inds[-1]+1
    cmax = occupied_col_inds[-1]+1
    return (rmin,cmin,rmax,cmax)

def image_crop(image,bbox=None,background='k'):
    """Automatically crop a masked image, slicing away the mask. Or crop to
    provided bounding box.

    Parameters
    ----------
    image : ndarray
        Input mask or masked image.
    bbox : tuple, optional
        Allows a custom bounding box to be input. Must be in form
        (min_row,min_col,max_row,max_col), where all values are ints.
        Otherwise, `bounding_box()` is used to calculate `bbox`.
    background : ('k'|'w'), optional (default 'k')
        Used if image is shape (h,w,3). Specifies background color of
        input image. 'k' for black or 'w' for white.
        Passed to `bounding_box()`.

    Returns
    -------
    output : ndarray
        Image, cropped to bounding box.
    """

    #bbox should be in format (min_row, min_col, max_row, max_col)
    h,w = image.shape[:2]
    if bbox is None:
        bb = bounding_box(image,background=background) #Get bbox
    elif len(bbox)==4:
        bb = bbox
    else:
        raise RuntimeError('Image crop error: `bbox` form not understood.')

    # Slice image to bb (while preventing out-of-bounds slicing)
    return image[bb[0]:min(bb[2],h),bb[1]:min(bb[3],w)]

def sort_mask_regions(mask):
    """Splits mask by region, returning a sorted list of single-region masks.

    Parameters
    ----------
    mask : bool array, shape (h,w)
        Boolean mask containing 2 objects

    Returns
    -------
    output : list of bool arrays
        List of single-region Boolean masks, ordered from upper-most to
        lower-most object in `mask`
    """
    labeled_mask = label(mask) #Label objects in bool image
    labels = np.unique(labeled_mask)
    labels = labels[1:] # drop region `0` (= image background)

    # get centroids for each region (returned as ``(row,col)``)
    props = regionprops(labeled_mask) # region properties
    centroids = [region['centroid'] for region in props]
    # get y-value for each region's centroid
    ys = [y for y,x in centroids]
    # sort region labels by their y-value
    # (sorted upper-most region to lower-most)
    ordered_labels = labels[np.argsort(ys)]

    # use that order to make a sorted list of single-region masks
    output = []
    for lbl in ordered_labels:
        temp = np.zeros(mask.shape,dtype=bool) # create empty mask
        temp[labeled_mask==lbl] = True # fill specific region
        output.append(temp)
    return output

def apply_mask(image,mask,background='k'):
    """Safely masks an image.

    Parameters
    ----------
    image : unint8 array, shape (h,w,3)
        Input image.
    mask : bool array, shape (h,w)
        Boolean mask.
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.

    Returns
    -------
    masked_image : ndarray
        Masked image where black pixels in `mask` replace those corresponding
        pixels in `image`.
    """

    masked_image = cv2.bitwise_and(image,image,mask=mask.astype(np.uint8)*255)

    if background is 'k':
        return masked_image
    elif background is 'w':
        masked_image[~mask] = (255,255,255)
        return masked_image
    else:
        raise ValueError('`background` value not understood.')

def reorient_wing_image(image,cutoff=0.05,background='k'):
    """Reorients an object in an image so that its top side is horizontal.

    Parameters
    ----------
    image : uint8 array, shape (h,w,3)
        Input image (mask) or masked image containing a single object.
    cutoff : float, optional
        Object (i.e. wing) in mask is clipped, longitudinally on either side by
        this amount before reorientation. This reduces the effect of artifacts
        at the ends of the object.
    background : ('k'|'w'), optional (default 'k')
        Color of `image`'s background. 'k' for black or 'w' for white.

    Returns
    -------
    output : ndarray
        An image containing the object in *image* that has been reoriented and cropped.
    """

    # Use `cutoff` to get start and end columns
    w = image.shape[1]
    start = int(np.floor(w * cutoff))-1 #Starting col after clipping
    end = int(np.ceil(w * (1-cutoff))) #Ending col after clipping

    # Convert image to grayscale
    m = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    m = m.T # Transpose image so we can look at columns first, then rows

    if background is 'k': # if background is black
        bg = 0; func = max
    elif background is 'w': # if background is black
        bg = 255; func = min
    else:
        raise ValueError('`background` value not understood.')

    # Find optimal rotation angle
    tops = []
    for col in range(start,end,10): #For every 10th column...
        if func(m[col])!=bg: # if col contains non-background pixel
            templist = []
            # take index of first non-background pixel & append to tops
            for row,val in enumerate(m[col]):
                if val!=bg:
                    templist.append([col,row])
                else:
                    continue
                tops.append(templist[0])
        else:
            continue

    xs,ys = np.transpose(tops)

    slope,_ = np.polyfit(xs,ys,1) #Fit line to top coordinates
    slope = np.rad2deg(np.arctan(slope)) #Convert slope to degrees

    #Rotate image:
    rotated = nd.rotate(image,slope,mode='constant',cval=bg)
    rotated = image_crop(rotated,background=background)
    return rotated

def pad_and_stack(image_list,height=256,background='k'):
    """Pad images to `height`, then assemble with one above the other.

    Parameters
    ----------
    image_list : list of arrays
        List of two uint8 images, each of shape (h,w,3).
    height : int, optional
        Target height (in px) of each image after padding bottom edge with
        background pixels. Default is 256.
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.

    Returns
    -------
    output : uint8 array
        Image formed by padding the bottom of each image with background pixels
        and combining them so that the second image is below the first.

    Raises
    ------
    RuntimeError : if an image's width:height is less than 2. This typically
    catches masking errors.
    """
    if background is 'k': # if background is black
        bg = 0
    elif background is 'w': # if background is black
        bg = 255
    else:
        raise ValueError('`background` value not understood.')

    # pad the bottom of each image with background pixels
    padded_list = []
    for image in image_list:
        h,w = image.shape[:2]
        if h > height:
            raise RuntimeError('Mask aspect ratio is throwing off padding. Check mask.')
        pad_width = ((0,height-h),(0,0),(0,0))
        padded = np.pad(image,pad_width,mode='constant',constant_values=bg)
        padded_list.append(padded)

    return np.vstack(padded_list)

# STANDARDIZER CLASS============================================================

# By using this class, the scaler/clf models only have to be loaded once for
# standardizing multiple specimen images.
class Standardizer:
    """For transforming an image or pair of images into a standardized square
    image.

    Parameters
    ----------
    scaler_fp : str
        Filepath to pickle of pre-trained sklearn.preprocessing.StandardScaler()
        object
    clf_fp : str
        Filepath to pickle of sklearn classifier, pre-trained to classify
        transformed pixels of `image` as either True (for foreground pixels)
        or False (for background pixels)
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.
    convex_hull : bool, optional (default True)
        Whether to apply `skimage.morphology.convex_hull_image()` to wing
        masks before returning them. Passed to `mask_wings()`.
    bgr : bool, optional (default False)
        Whether to expect `image` channels to be in order BGR (as from
        cv2.imread()). Otherwise, channels assumed to be RGB. If bgr, returns
        BGR image, otherwise returns RGB.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.StandardScaler object
        Instance of a pre-trained StandardScaler object.
    clf_ : sklearn classifier object
        Instance of a pre-trained classifier object that accepts pixel-wise
        features vectors from `transparency_transform(image)` and outputs True
        for pixels that are predicted to be from a wing and False for background
        pixels.

    Methods
    -------
    make_square : builds a standardized square from an image or pair of images

    Examples
    --------
    import autoID,cv2
    scaler_fp = 'path/to/file/scaler.pkl'
    clf_fp = 'path/to/file/clf.pkl'
    squarer = autoID.Standardizer(scaler_fp,clf_fp,background='k',bgr=True)
    img = cv2.imread('path/to/file/image_with_2_wings.tif') # read img as BGR
    square = squarer.make_square(img) # convert to standardized square
    """

    def __init__(self,scaler_fp,clf_fp,background='k',convex_hull=True,
                 bgr=False):
        self.background     = background
        self.convex_hull    = convex_hull
        self.bgr            = bgr

        # load models
        self.scaler_         = _load_model(scaler_fp)
        self.clf_            = _load_model(clf_fp)

    def mask_wings(self,image,n_wings=2):
        """Detect & mask wings in an image.

        Steps: (1) transform image to a get 6-len feature vector per pixel
               (2) apply trained classifier to predict whether each pixel is a wing
               (3) fill in holes in predicted mask
               (4) filter out all but largest `n_wings` regions in mask
               (5) optionally apply convex_hull_image to each kept region

        Parameters
        ----------
        image : uint8 array, shape (h,w,3)
            Input image. Channels must be ordered RGB (not cv2's BGR)!
        n_wings : int, optional (default 2)
            The number of wings to find and mask in `image`

        Returns
        -------
        mask : bool array, shape (h,w)
            Image mask, where `n_wings` objects (wings) are True and background
            pixels are False

        Raises RuntimeError if the number of recovered regions < n_wings.
        """
        scaler      = self.scaler_
        clf         = self.clf_

        batch_size_limit = 50000 # lower if raises MemoryError

        if self.bgr: # convert image from BGR -> RGB
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Transform image to features for each pixel
        img_trans = transparency_transform(image)
        h,w,d = img_trans.shape # dims of transformed image

        # Flatten to shape (n_pixels,n_features)
        pixels = img_trans.reshape((h*w,d))

        # Predict whether each pixel is a wing
        batch_size = int(batch_size_limit/d)
        if len(pixels)>batch_size: # work in batches to prevent MemoryError
            # NOTE: uint8->float64 makes scaler transform memory intensive

            divs = int(len(pixels)/batch_size) # `divs`+1 batches will be used
            predicted = np.zeros((h*w),dtype=bool) # empty array to hold prediction

            for div in range(divs+1):
                if div<divs: # work with all but last batch
                    batch = pixels[batch_size*div:batch_size*(div+1)]
                    batch = batch.astype(np.float64) # cast to float for scaler
                    batch = scaler.transform(batch)
                    predicted[batch_size*div:batch_size*(div+1)] = clf.predict(batch)

                elif (len(pixels)%batch_size)>0: # last batch, if any remaining
                    batch = pixels[batch_size*div:batch_size*(div+1)]
                    batch = batch.astype(np.float64) # cast to float for scaler
                    batch = scaler.transform(batch)
                    predicted[-(len(pixels)%batch_size):] = clf.predict(batch)

        else: # if image is small, predict in a single go
            predicted = clf.predict(pixels)

        # Reshape back to image dims
        predicted_mask = predicted.reshape((h,w))

        # Do morphological hole filling
        filled_mask = nd.binary_fill_holes(predicted_mask) #Hole filling

        # Find regions in image and determine which to keep
        labeled_mask = label(filled_mask) #Label objects in bool image
        props = regionprops(labeled_mask) # region properties
        areas = [region['area'] for region in props]
        labels = np.unique(labeled_mask)
        labels = labels[1:] # drop region `0` (= image background)
        if len(labels)<n_wings: # catch if there aren't enough labeled regions
            raise RuntimeError('An insufficient number of objects was detected in image.')
        # keep only the `n_wings` regions that have the largest pixel area
        labels_to_keep = labels[np.argsort(areas)][-n_wings:]

        # Fill in empty mask with only the regions in labels_to_keep
        mask = np.zeros(labeled_mask.shape,dtype=int)
        for lbl in labels_to_keep:
            mask[labeled_mask==lbl] = lbl

        # Optionally, apply convex_hull_image to each region in mask
        if self.convex_hull:
            for lbl in labels_to_keep:
                temp = np.zeros(mask.shape,dtype=int)
                temp[mask==lbl] = 1
                temp = convex_hull_image(temp) #Compute hull image
                mask[temp==1] = lbl

        return mask.astype(bool)


    def make_square(self, image1, mask1=None, image2=None, mask2=None,
                    flip=False, switch=False):
        """Make a standardized square image from 1 two-winged image or 2
        one-winged images.

        Parameters
        ----------
        image1,image2 : uint8 array
            Input images, each containing 1 wing. Can be color or grayscale.
        mask1,mask2 : bool array, optional
            Allows custom Boolean masks to be submitted for image1 &/or image2.
            If not provided, `mask_wings()` is used to get mask.
        background : ('k'|'w'), optional (default 'k')
            Desired background color for standardized square.
            'k' for black or 'w' for white.
        bgr : bool, optional (default False)
            Whether channels of input images are BGR (as from `cv2.imread()`).
            Otherwise, images are assumed to be RGB. If bgr, returns
            BGR image, otherwise returns RGB. Passed to `mask_wings()`
        convex_hull : bool, optional (default True)
            Whether to apply `skimage.morphology.convex_hull_image()` to wing
            masks before returning them. Passed to `mask_wings()`.
        flip : bool,optional (default False)
            Option to flip images left-to-right.
        switch : bool,optional (default False)
            Option to switch the order of the first & second image.

        Returns
        -------
        output : uint8 array, shape (512,512,3)
            Standardized 512-px square image, where the fore- and hindwings are
            masked, reoriented, resized and placed in the upper and lower halves
            of the square, respectively. If `bgr=True`, image is returned as
            BGR, otherwise returned as RGB.

        Raises
        ------
        RuntimeError : if mask_wings() fails to find enough objects in image(s)
        """
        bg          = self.background

        # if wings are both contained in one image
        if image2 is None: # image1 contains 2 wings
            if mask1 is None: # get mask for image1
                mask1 = self.mask_wings(image1,n_wings=2)
            elif not _check_image_mask_match(image1,mask1):
                raise ValueError('`image1` and `mask1` are not the same shape.')

            sorted_masks = sort_mask_regions(mask1)
            if switch:
                sorted_masks = sorted_masks[::-1]
            masked_images = list(map(lambda y: apply_mask(image1,y,background=bg),
                                sorted_masks))

        # if wings are seperated between 2 images
        elif image2 is not None: # image1 and image2 each contain a wing
            if mask1 is None:
                mask1 = self.mask_wings(image1,n_wings=1)
            elif not _check_image_mask_match(image1,mask1):
                raise ValueError('`image1` and `mask1` are not the same shape.')

            if mask2 is None:
                mask2 = self.mask_wings(image2,n_wings=1)
            elif not _check_image_mask_match(image2,mask2):
                raise ValueError('`image2` and `mask2` are not the same shape.')

            if switch:
                images = [image2,image1]
                sorted_masks = [mask2,mask1]
            else:
                images = [image1,image2]
                sorted_masks = [mask1,mask2]
            masked_images = list(map(lambda x,y: apply_mask(x,y,background=bg),
                                images,sorted_masks))

        cropped_images = list(map(lambda x: image_crop(x,background=bg),
                             masked_images))
        rotated_images = list(map(lambda x: reorient_wing_image(x,background=bg),
                             cropped_images))
        resized_images = list(map(lambda x: resize(x,width=512),rotated_images))
        square = pad_and_stack(resized_images,background=bg)

        if flip:
            square = np.fliplr(square)

        if self.bgr:
            return square[:,:,::-1]
        else:
            return square
