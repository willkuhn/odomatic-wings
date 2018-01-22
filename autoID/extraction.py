# -*- coding: utf-8 -*-
# Copyright (C) 2017 William R. Kuhn

import os
import numpy as np
from scipy import signal
import skimage.exposure, skimage.color, skimage.segmentation
import cv2
from autoID.preprocessing import sort_mask_regions

from skimage.feature import canny # for older version of _parabolaParams
#import matplotlib.pyplot as plt
#import skimage.morphology as morph

"""
===============================================================================
CHROMATIC SAMPLING
===============================================================================
"""
def _safe_div(num,denom): #safely do num/denom when denom contains zeros
    d = denom.copy()
    d[d==0.] = np.inf
    return num/d

def RGB2chrom(image):
    """Convert an RGB image pixel values to chromatic coordinates.

    Parameters
    ----------
    image : ndarray
        RGB input image

    Returns
    -------
    output : ndarray
        3-channel image with transformed pixel values.

    Notes
    -----
    Floating point (0.0-1.0) RGB color pixel values (R,G,B) are converted to
    chromatic coordinates (r,b,g) with the transformation:
        r = R / (R+G+B)
        g = G / (R+G+B)
        b = B / (R+G+B)

    References
    ----------
    [1] Gillespie AR, Kahle AB, Walker RE (1987) Color enhancement of highly
        correlated images. II. Channel ratio and “chromaticity” transformation
        techniques. Remote Sens Environ 22: 343–365.
        doi:10.1016/0034-4257(87)90088-5.
    [2] Woebbecke DM, Meyer GE, Von Bargen K, Mortensen DA (1995) Color indices
        for weed identification under various soil, residue, and lighting
        conditions. Trans ASAE 38: 259–269.
    [3] Sonnentag O, Hufkens K, Teshera-Sterne C, Young AM, Friedl M, Braswell
        BH, et al. (2012) Digital repeat photography for phenological research
        in forest ecosystems. Agric For Meteorol 152: 159–177.
        doi:10.1016/j.agrformet.2011.09.009.
    """
    try:
        img = image.astype(np.float)
        chans = cv2.split(img) #Split image into channels
        tot = sum(chans) #Sum up pixel values across channels
        chans_new = list(map(lambda channel: _safe_div(channel,tot),chans)) #transform channels
        new = cv2.merge(chans_new) #merge transformed channels
        new[np.isnan(new)] = 0. #convert any pixel values where x/0. (NaNs) to 0.
        new *= 255.0/new.max() #convert pixel values from (0-1.) to (0.-255.)
        new = new.astype(np.uint8)
        return new
    except:
        print ("Error during conversion to chromaticity.")

def chrom_sample(image,use_chromaticity=True):
    """Converts *image* to coefficients from the mean & stdev of pixel values
    of different subdivisions of each image channel.

    Version 2.

    Parameters
    ----------
    image : ndarray
        Input image of shape (512,512,3).
    use_chromaticity : bool, optional
        If True, RGB image is transformed to chromatic coordinates prior to
        extraction. If False, untransformed RGB image is used.

    Returns
    -------
    output : list
        List of 138 coefficients (46 for each channel)

    Notes
    -----
    In this process, the mean and stdev of pixel values in each color channel
    are calculated, then the image is broken into halves (row-wise) and the
    calculations are repeated on those.  The process is repeated on the thirds
    (column-wise) of those halves (sixths), and the halves (row- wise) of those
    sixths (12ths). If color channels are r,g,b, the order of resultant coeff-
    icients is [whole img mean (1), halves means (2), fourths means (4), 16ths
    means (16), whole image stdevs (1), halves stdevs (2), fourths stdevs (4),
    16ths stdevs(16)] for r (46), then g (46), then b (46), all returned as a
    flat list.

    References
    ----------
    [1] Le-Qing, Z. & Zhen, Z. (2012) Automatic insect classification based on
        local mean colour feature and Supported Vector Machines. Oriental
        Insects 46, 260–269.

    TODO: try array slicing instead of array_split to get image regions (faster?)
    """
    if use_chromaticity:
        image = RGB2chrom(image) #Convert img to chromatic coordinates
    chans = np.transpose(image,(2,0,1)).astype(np.float) #Split channels & cast to float
    #For each element in list of arrays *i*, flatten array & apply *func*
    coeffs = np.array([])
    for ch in chans: #For each channel...
        #Get channel parts:
        h0 = np.array_split(ch,2,axis=0) #Halves (split row-wise)
        h1 = list(map( lambda x: np.array_split(x,2,axis=1), h0)) #halves of halves
        h1 = np.concatenate(h1) #4ths
        h2 = list(map( lambda x: np.array_split(x,2,axis=0), h1)) #halves of 4ths
        h2 = np.concatenate(h2) #(8ths)
        h2 = list(map( lambda x: np.array_split(x,2,axis=1), h2)) #halves of 8ths
        h2 = np.concatenate(h2) #16ths

        #Means pixel values of channel parts:
        ch_m = [ch.ravel().mean()] #Overall channel mean
        h0_m = [i.ravel().mean() for i in h0] #Means of halves
        h1_m = [i.ravel().mean() for i in h1] #Means of 4ths
        h2_m = [i.ravel().mean() for i in h2] #Means of 16ths

        #Stdevs of pixel values of image parts:
        ch_s = [ch.ravel().std()] #Overall channel stdev
        h0_s = [i.ravel().std() for i in h0] #Stdevs of halves
        h1_s = [i.ravel().std() for i in h1] #Stdevs of 4ths
        h2_s = [i.ravel().std() for i in h2] #Stdevs of 16ths

        ch_coeffs = np.concatenate((ch_m,h0_m,h1_m,h2_m,ch_s,h0_s,h1_s,h2_s))
        #print (ch_coeffs.shape)
        coeffs = np.append(coeffs,ch_coeffs)

    return coeffs.reshape([-1])


def chrom_sample_v1(image):
    """OLD PROCEDURE. Converts *image* to coefficients from the mean & stdev of
    pixel values of different subdivisions of each image channel.

    Parameters
    ----------
    image : ndarray
        Square, 3-channel input image
        Image is presumed to be chromatic-transformed using RGB2chrom() and a
        512 x 512 px square

    Returns
    -------
    output : list
        List of 126 coefficients (42 for each channel)

    Notes
    -----
    In this process, the mean and stdev of pixel values in each color channel
    are calculated, then the image is broken into halves (row-wise) and the
    calculations are repeated on those.  The process is repeated on the thirds
    (column-wise) of those halves (sixths), and the halves (row- wise) of those
    sixths (12ths). If color channels are r,g,b, the order of resultant coeff-
    icients is [whole img mean (1), halves means (2), sixths means (6), 12ths
    means (12), whole image stdevs (1), halves stdevs (2), sixths stdevs (6),
    12ths stdevs(12)] for r (42), then g (42), then b (42), all returned as a
    flat list.

    References
    ----------
    [1] Le-Qing, Z. & Zhen, Z. (2012) Automatic insect classification based on
        local mean colour feature and Supported Vector Machines. Oriental
        Insects 46, 260–269.
    """
    image = RGB2chrom(image) #Convert img to chromatic coordinates
    chans = cv2.split(image.astype(np.float)) #Split img to channels (as float)
    #TODO ***mapflat needs optimization here; costs majority of run time***
    mapflat = lambda func,i: list(map(lambda x: func([n for e in x for n in e]), i))
    #For each element in list of arrays *i*, flatten array & apply *func*
    coeffs = np.array([])
    for ch in chans: #For each channel...
        #Get channel parts:
        h0 = np.array_split(ch,2,axis=0) #Halves
        h1 = list(map( lambda x: np.array_split(x,3,axis=1), h0))
        #******Issue with following statement: arrays not all same shape*******
        h1 = np.concatenate(h1) #Sixths
        h2 = list(map( lambda x: np.array_split(x,2,axis=0), h1))
        h2 = np.concatenate(h2) #12ths

        #Means pixel values of channel parts:
        ch_m = mapflat(np.mean,[ch]) #Overall channel mean
        h0_m = mapflat(np.mean,h0) #Means of halves
        h1_m = mapflat(np.mean,h1) #Means of 6ths
        h2_m = mapflat(np.mean,h2) #Means of 12ths

        #Stdevs of pixel values of image parts:
        ch_s = mapflat(np.std,[ch]) #Overall channel stdev
        h0_s = mapflat(np.std,h0) #Stdevs of halves
        h1_s = mapflat(np.std,h1) #Stdevs of 6ths
        h2_s = mapflat(np.std,h2) #Stdevs of 12ths

        ch_coeffs = np.concatenate((ch_m,h0_m,h1_m,h2_m,ch_s,h0_s,h1_s,h2_s))
        #print (ch_coeffs.shape)
        coeffs = np.append(coeffs,ch_coeffs)

    return coeffs.reshape([-1])


"""
===============================================================================
SAMPLING WITH GABOR WAVELET TRANSFORMATION
===============================================================================
"""
def _gabor_filterfft(image,frequency,theta=0,bandwidth=1,sigma_x=None,sigma_y=None, offset=0, mode="same"):
    """Exactly the same as skimage.filters.gabor_filter() except uses
    scipy.signal.fftconvolve() (not ndimage.convolve()) to run convolution,
    which prevents 'Memory Error'. See scipy.signal.fftconvolve() for
    explanation of parameters.

    Parameters
    ----------
    image : ndarray
        Input image
    others : ...

    Returns
    -------
    output : ndarray
        Filtered image, same shape as *image*.
    """
    g = skimage.filters.gabor_kernel(frequency, theta, bandwidth, sigma_x, sigma_y, offset)
    filtered_real = signal.fftconvolve(image, np.real(g), mode=mode)
    filtered_imag = signal.fftconvolve(image, np.imag(g), mode=mode)
    return filtered_real, filtered_imag

def gabor_sample(image,c,a=3):
    """Converts *image* to coefficients from Gabor wavelet transformation.

    Parameters
    ----------
    image : ndarray
        Square, RGB or grayscale image
        If RGB, image is converted to grayscale before sampling
        Image is presumed to be a 512 x 512 px square
    c : int
        Number of wavelets on each side of square
        E.g. c=2 places 2x2=4 1/4-image-sized wavelets on image; c=4 gives 16
        Typically 1, 2, 4, 8, or 16
    a : int, optional
        Number of angles (theta) for wavelets, evenly spaced between 0 & 2pi
        E.g. a=3 gives 3 angles: theta = 0, 2pi/3, & 4pi/3
        Default is 3

    Returns
    -------
    output : list
        List of c*c*a*2 coefficients

    Notes
    -----
    In this process, image is converted to grayscale, and a c x c grid of
    circles (with diameter image_side_length/c) is set up on the square image.
    One of several wavelets is convolved over the image and values at the
    centers of the circles are taken to as Gabor coefficients.  The real and
    imaginary component of the wavelet at each of *a* angles is used, so a*2
    wavelets are convolved over the image and c*c*a*2 coefficients are taken.

    References
    ----------
    [1] Russell, K. N., Do, M. T., Huff, J. C. & Platnick, N. I. (2007)
        Introducing SPIDA-web: wavelets, neural networks and internet access-
        ibility in an image-based automated identification system. pp 131–152.
        in MacLeod, N., ed. Automated Object Identification in Systematics:
        Theory, Approaches, and Applications, CRC Press.
    [2] Adapted from Mathematica script by Gareth Russell, NJIT

    """
    if image.ndim == 3: #If image is color, convert to grayscale
        image = skimage.color.rgb2gray(image)
    a = float(a)
    c = float(c)
    angleList = np.arange(0, 1 - 1/a, 1/a) * 2*np.pi #Get angles list
    n = float(image.shape[0]) #Length of side of square
    freq = 0.05*32/n #Get appropriate wavelet frequency
    g = n/c
    scaleCenters = np.round(np.linspace(0.,n-g,num=c) + g/2.).astype(int)
    scaleCenters = [(y,x) for y in scaleCenters for x in scaleCenters] #Get coords for centers
    sigma_x,sigma_y = np.asarray([g,g])/2.

    coeffs_list = []
    for theta in angleList:
        re,im = _gabor_filterfft(image,freq,theta=theta, sigma_x=sigma_x,sigma_y=sigma_y) #Get Gabor arrays
        #Extract coord centers from Gabor arrays:
        re_c = [re[i,j] for i,j in scaleCenters]
        im_c = [im[i,j] for i,j in scaleCenters]

        coeffs_list.append(re_c)
        coeffs_list.append(im_c)
    coeffs_list = np.concatenate(coeffs_list)
    return coeffs_list


"""
===============================================================================
WING MORPHOMETRY
===============================================================================
"""
def _masks_from_square(square,background='k'):
    """Recover individual wing masks from a standardized square image.


    Parameters
    ----------
    square : ndarray
        Square, RGB or grayscale image.
        Image is presumed to be a 512 x 512 px square and contain 2 objects.
    background : ('k'|'w'), optional (default 'k')
        Background color of standardized square.
        'k' for black or 'w' for white.

    Returns
    -------
    output : list of bool arrays
        List of single-region Boolean masks, ordered from upper-most to
        lower-most object in `square`
    """
    if square.ndim == 3:
        m = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #Binarize image:
    if background is 'k':
        mask = m>0
    elif background is 'w':
        mask = m<255
    else:
        raise ValueError('`background` value not understood.')

    #Remove small objects?
    #min_size = ?
    #mask = morph.remove_small_objects(square_mask.astype(bool),
    #                                         min_size=min_size)

    return sort_mask_regions(mask)


def _wingAreaRatio(sorted_masks):
    """Calculates area ratio of object in 1st mask to object in 2nd mask.
        Note: since each wing has been resized to fit the width of the
        standardized square image, this proportion is not true to life;
        however, it is comparable among specimens that have undergone this
        standardization procedure.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (1 value)
        Area ratio (i.e. forewing_area : hindwing_area)
    """
    props = list(map(skimage.measure.regionprops,sorted_masks))
    areas = list(map(lambda p: p[0]['area'],props))
    return [ areas[0]/float(areas[1]) ]

def _wingElognation(sorted_masks):
    """Calculates elongation of objects in 1st & 2nd mask.
    See skimage.measure.regionprops() for explanation of 'elongation'.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (2 values)
        Elongation of object in each mask.
    """
    props = list(map(skimage.measure.regionprops,sorted_masks))
    widths = np.asarray(map(lambda p: p[0]['minor_axis_length'],props))
    lengths = np.asarray(map(lambda p: p[0]['major_axis_length'],props))
    return list( 1 - widths/lengths )

def _antePostRatio(sorted_masks):
    """For each mask, gives area ratio of upper half to lower half. Halves
        are determined by the position of the wing's centroid.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (2 values)
        Area ratio for each mask.
    """
    props = list(map(skimage.measure.regionprops,sorted_masks))
    cents = list(map(lambda p: int(p[0]['centroid'][0]),props))
    ret = []
    for i in range(2):
        sAnt = skimage.measure.regionprops(sorted_masks[i][:cents[i]])
        sPost = skimage.measure.regionprops(sorted_masks[i][cents[i]+1:])
        sAnt_area = sAnt[0]['area']
        sPost_area = sPost[0]['area']
        ret.append(sAnt_area/float(sPost_area))
    return ret

def _proxDistRatio(sorted_masks):
    """For each mask, gives area ratio of left half to right half. Halves
        are determined by the position of the wing's centroid.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (2 values)
        Area ratio for each mask.
    """
    props = list(map(skimage.measure.regionprops,sorted_masks))
    cents = list(map(lambda p: int(p[0]['centroid'][1]),props))
    ret = []
    for i in range(2):
        sProx = skimage.measure.regionprops(sorted_masks[i][:,:cents[i]])
        sDist = skimage.measure.regionprops(sorted_masks[i][:,cents[i]+1:])
        sProx_area = sProx[0]['area']
        sDist_area = sDist[0]['area']
        ret.append(sProx_area/float(sDist_area))
    return ret

def _regressionOfThickness(sorted_masks):
    """For each mask, gives slope of linear regression of mask thickness by
    column.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (2 values)
        Slope for each mask.
    """
    ss = []
    for mask in range(2): #For each mask

        # count number of 1s in each column (i.e. number of white px)
        thicknesses = [(i+1,len(c[c==1])) for i,c in enumerate(sorted_masks[mask].T)]

        # fit a line to thickness over column number
        x,y = np.transpose(thicknesses)
        s,b = np.polyfit(x,y,1) #Fit a line to thicknesses

        # visualize
        #plt.plot(x,y) # plot thickness
        #plt.plot(x,x*s+b,c='r') # plot fit line

        ss.append(s)
    return ss #Return slope of fit line

def _widestColumn(sorted_masks):
    """For each mask, finds the column that contains the most white pixels
        (i.e. the 'thickest' part of the wing) and divides it by the total
        width of the mask. Thus, for each wing, it gives a proportion of where
        the wing is thickest. Closer to 0 mean thickest basally, closer to
        1 means thickest distally.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (2 values)
        Proportion for each mask.
    """
    ret = []
    for mask in range(2):
        w = float(sorted_masks[mask].shape[1])

        # count number of 1s in each column (i.e. number of white px)
        thicknesses = [len(c[c==1]) for c in sorted_masks[mask].T]

        # Find the column with the most pixels
        widest_col = np.argmax(thicknesses) # thickest column(s)
        widest_col = np.mean(widest_col) # in case argmax returns >1 value

        # Divide that by width and append to ret
        ret.append(widest_col / w)
    return ret

def _parabolaParams(sorted_masks):
    """For each mask, gives x^3 coefficient for 3rd-order polynomial
    (d in y = dx^3 + cx^2 + bx + a) fit to curvature of upper and lower half
    of the right 40% of mask (i.e. coefficients that describe the
    curvature of the wingtips).

    Note: this script uses cv2.Canny, which seems less accurate, but is 10X
    faster than skimage.feature.canny.

    Version 2.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (4 values)
        Upper and lower coefficients for each mask in 1 flattened list.
    """
    ret = []
    for i in range(2): # for each mask (i.e. each wing)
        w= float(sorted_masks[i].shape[1]) # mask width
        blur = cv2.blur(cv2.convertScaleAbs(sorted_masks[i]),(1,1))
        edges = cv2.Canny(blur,0.1,.2,1)
        edges = edges.astype(int) # important: "labels" pixels (label=1 for all edge px)

        # get tip of wing (distal 40% of wing)
        tip = edges[:,-int(w*0.4):]

        # find row-wise middle of wing using the extreme tip (distal 2% of wing)
        extrtip = edges[:,-int(w*0.02):]
        extrtip_coords = skimage.measure.regionprops(extrtip)[0]['coords']
        middle = int(extrtip_coords[:,0].mean())
        #print (middle)

        # fit 3rd order polynom to coords of all px in upper half if wing tip
        up = tip[:middle]
        up_coords = skimage.measure.regionprops(up)[0]['coords']
        yu,xu = np.transpose(up_coords)
        fitu = np.polyfit(xu,yu,3)

        # fit 3rd order polynom to coords of all px in lower half if wing tip
        down = tip[middle:]
        down_coords = skimage.measure.regionprops(down)[0]['coords']
        yd,xd = np.transpose(down_coords)
        fitd = np.polyfit(xd,yd,3)

        # get value for `d` from `fitu` and `fitd`, and append to ret
        ret.append(fitu[0])
        ret.append(fitd[0])
    return ret

def _parabolaParams_v1(sorted_masks):
    """For each mask, gives x^3 coefficient for 3rd-order polynomial
    (d in y = dx^3 + cx^2 + bx + a) fit to curvature of upper and lower half
    of the right 40% of mask (i.e. coefficients that describe the
    curvature of the wingtips).

    Note: Version 1. This script uses skimage.feature.canny, which is 10X slower than
    cv2.Canny.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (4 values)
        Upper and lower coefficients for each mask in 1 flattened list.
    """
    ret = []
    for i in range(2): # for each mask (i.e. each wing)
        w= float(sorted_masks[i].shape[1]) # mask width
        edges = canny(sorted_masks[i].astype(np.float)) # wing edges
        edges = edges.astype(int) # important: "labels" pixels (label=1 for all edge px)

        # get tip of wing (distal 40% of wing)
        tip = edges[:,-int(w*0.4):]

        # find row-wise middle of wing using the extreme tip (distal 2% of wing)
        extrtip = edges[:,-int(w*0.02):]
        extrtip_coords = skimage.measure.regionprops(extrtip)[0]['coords']
        middle = int(extrtip_coords[:,0].mean())
        #print (middle)

        # fit 3rd order polynom to coords of all px in upper half if wing tip
        up = tip[:middle]
        up_coords = skimage.measure.regionprops(up)[0]['coords']
        yu,xu = np.transpose(up_coords)
        fitu = np.polyfit(xu,yu,3)

        # fit 3rd order polynom to coords of all px in lower half if wing tip
        down = tip[middle:]
        down_coords = skimage.measure.regionprops(down)[0]['coords']
        yd,xd = np.transpose(down_coords)
        fitd = np.polyfit(xd,yd,3)

        # get value for `d` from `fitu` and `fitd`, and append to ret
        ret.append(fitu[0])
        ret.append(fitd[0])
    return ret

def _parabolaParams_old2(sorted_masks):
    """For each mask, gives x^3 coefficient for 3rd-order polynomial
    (d in y = dx^3 + cx^2 + bx + a) fit to curvature of upper and lower half
    of the right 40% of mask (i.e. coefficients that describe the
    curvature of the wingtips).

    Note: this script uses both cv2.Canny (faster than skimage.feature.canny)
    and only runs skimage.measure.regionprops once per loop, rather than twice.

    Parameters
    ----------
    sorted_masks : list
        Output from _masks_from_square(). Assumption: 2 same-shaped ndarray masks.

    Returns
    -------
    output : list (4 values)
        Upper and lower coefficients for each mask in 1 flattened list.
    """
    ret = []
    for i in range(2): # for each mask (i.e. each wing)
        w= float(sorted_masks[i].shape[1]) # mask width
        blur = cv2.blur(cv2.convertScaleAbs(sorted_masks[i]),(1,1))
        edges = cv2.Canny(blur,0.1,.2,1)
        edges = edges.astype(int) # important: "labels" pixels (label=1 for all edge px)
        coords = skimage.measure.regionprops(edges)[0]['coords']

        # get tip of wing (distal 40% of wing)
        tip = coords[coords[:,1]>int(w*0.6)]

        # find row-wise middle of wing using the extreme tip (distal 2% of wing)
        extrtip = coords[coords[:,1]>int(w*0.98)]
        middle = int(extrtip[:,0].mean())
        #print (middle)

        # fit 3rd order polynom to coords of all px in upper half if wing tip
        up = tip[tip[:,0]<middle]
        yu,xu = np.transpose(up)
        fitu = np.polyfit(xu,yu,3)

        # fit 3rd order polynom to coords of all px in lower half if wing tip
        down = tip[tip[:,0]>=middle]
        yd,xd = np.transpose(down)
        fitd = np.polyfit(xd,yd,3)

        # get value for `d` from `fitu` and `fitd`, and append to ret
        ret.append(fitu[0])
        ret.append(fitd[0])
    return ret

def morphometric_sample(square,sorted_masks=None):
    """Samples morphometric information from *square*.

    Parameters
    ----------
    square : array
        Square image from autoID.preprocessing.make_square.
    sorted_masks : list, optional
        Output from _masks_from_square(). Allows this allows pre-computed masks
        to be used here.

    Returns
    -------
    output : tuple (15 values)
        Output in this order [war,we1,we2,apr1,apr2,pdr1,pdr2,rot1,rot2,wc1,
        wc2,pp1,pp2,pp3,pp4] where war=wingAreaRatio, we=wingElongation,
        apr=antePostRatio, pdr=proxDistRatio, rot=regressionOfThickness,
        wc=widestColumn, & pp=parabolaParams.
    """
    if sorted_masks is None:
        sorted_masks = _masks_from_square(square)

    l = len(sorted_masks)
    s0,s1 = sorted_masks[0].shape,sorted_masks[1].shape
    if l != 2:
        os.environ['error_message'] = 'Expecting 2 masks. Got %d.' % (len(sorted_masks))
        raise RuntimeError('Expecting 2 masks. Got %d.' % (len(sorted_masks)))
    elif s0 != s1:
        os.environ['error_message'] = 'Sorted masks not same shape: %s and %s.' % (str(s0),str(s1))
        raise RuntimeError('Sorted masks not same shape: %s and %s.' % (str(s0),str(s1)))

    a = _wingAreaRatio(sorted_masks)
    b = _wingElognation(sorted_masks)
    c = _antePostRatio(sorted_masks)
    d = _proxDistRatio(sorted_masks)
    e = _regressionOfThickness(sorted_masks)
    f = _widestColumn(sorted_masks)
    g = _parabolaParams(sorted_masks)
    return np.concatenate([a,b,c,d,e,f,g])
