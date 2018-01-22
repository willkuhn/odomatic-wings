# -*- coding: utf-8 -*-
# Copyright (C) 2015 William R. Kuhn

import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from prettytable import PrettyTable
#from . import preprocessing, extraction

def truncate_train_features(feature_set,k=10):
    """Truncates features in set with PCA if n_features > k, normalizes data,
    and returns both transformed features and PCA model, which can be applied
    to testing or novel samples.

    Parameters
    ----------
    feature_set : array
        Input data array of shape (n_samples,n_features)
    k : int
        Max number of features to be returned. Default is 10.

    Returns
    -------
    trans : array
        A normalized array of shape (n_samples,k) containing the first k principal
        components of feature_set if n_feature > k, or of shape (n_samples,
        n_features) otherwise. The array is normalized by row (sample).
    pca_fit : sklearn.decomposition.pca.PCA or None
        A PCA model to be used to transform test samples if n_feature > k, or
        None othersize.
    """
    if len(feature_set[0]) > k:
        pca = PCA(n_components = k)
        pca_fit = pca.fit(feature_set) #fit PCA to data
        trans = pca_fit.transform(feature_set) #transform data with fit
    else:
        pca_fit = None
        trans = feature_set #Don't transform data
    trans = normalize(trans,axis=1,copy=False)
    return trans,pca_fit

def truncate_test_features(feature_set,fitting):
    """Truncates features in set with PCA if their training counterparts were
    truncated, normalizes data, and returns the transformed set.

    Parameters
    ----------
    feature_set : array
        Input data array of shape (n_samples,n_features)
    fitting : sklearn.decomposition.pca.PCA or None
        A PCA model for transforming data, or None (i.e. the 2nd output from
        truncate_train_features()).
    Returns
    -------
    trans : array
        A normalized array transformed with fitting, or just a normalized
        version of feature_set if fitting == None.
    """
    if fitting == None:
        trans = feature_set #Don't transform data
    else:
        trans = fitting.transform(feature_set) #transform data with fit
    trans = normalize(trans,axis=1,copy=False)
    return trans

def id_from_coeffs(sample,featureLengths,fittings,clf_model,verbose='False'):
    fl = [sum(featureLengths[:i]) for i in range(1,len(featureLengths)+1)] #Accumulating list of feature lengths
    sample_sets = np.split(sample,fl[:-1]) #Split into feature sets
    try:
        sample_sets_trans = list(map(lambda x,y: truncate_test_features(x,y),(sample_sets),fittings)) #Truncate sets
    except: print ('Error occurred during coefficient ordination/truncation.'); return None
    sample_trans = np.hstack(sample_sets_trans)[0] #Merge sets
    try:
        pred = clf_model.predict_proba(sample_trans)[0] #Class predictions on data_test_trans
    except: print ('Error occurred during prediction.'); return None
    classes = clf_model.classes_ #List of classes in the model
    ranked_preds = [zip(classes,pred)[i] for i in list(np.argsort(pred))][::-1] #Ranked list of predicted labels
    if verbose in ['True',0]:
        #Make PrettyTable of results:
        results = PrettyTable(['Predicted ID','Score'])
        results.align ='l'
        results.float_format = '1.4'
        for i,j in ranked_preds:
            results.add_row([i,j])
        print (results)
        return ranked_preds
    else: return ranked_preds

def id_from_image(image,featureLengths,fittings,clf_model,verbose='False'):
    """Identifies an image (i.e. returns a ranked list of predicted class labels).
    Image is first preprocessed, features are extracted from it, feature sets are
    truncated with *fittings* & class predictions are made with *clf_model*.

    Parameters
    ----------
    sample : tuple
        List of extracted image features.
    featureLengths : tuple
        List of n_features for each feature set (e.g. (16,126,510))
    fittings : list
        Elements are used to truncate test feature sets.
        Elements should be sklearn.decomposition.pca.PCA or None.
    clf_model : sklearn model
        Used to predict class label of *image*.
    verbose : bool
        If 'True', ranked results are returned as PrettyTable object
    Returns
    -------
    trans : list
        List of class labels & prediction scores, sorted by score.
        First item in list is the top-ranked.
    """

    try:
        square = preprocessing.make_square(image)
    except: print ('Error occurred during image preprocessing.'); return None
    try:
        morphCoeffs = extraction.morphometric_sample(square)
        chromCoeffs = tuple(extraction.chrom_sample(square))
        gaborCoeffs = list(map(lambda x: extraction.gabor_sample(square,x,3),[8,4,2,1]))
        gaborCoeffs = tuple(np.concatenate(gaborCoeffs))

        sample = tuple(np.concatenate([morphCoeffs,chromCoeffs,gaborCoeffs]))
    except: print ('Error occurred during feature extraction.'); return None

#    print ('Original image:'); plt.imshow(image); plt.show()
#    print ('Standardized square image:'); plt.imshow(square);plt.show()
    return id_from_coeffs(sample,featureLengths,fittings,clf_model,verbose=verbose)

def id_from_square(square,featureLengths,fittings,clf_model,verbose='False'):
    """Identifies an image (i.e. returns a ranked list of predicted class labels).
    Image is first preprocessed, features are extracted from it, feature sets are
    truncated with *fittings* & class predictions are made with *clf_model*.

    Parameters
    ----------
    square : array
        Input image. Should be a standardized square image.
    featureLengths : tuple
        List of n_features for each feature set (e.g. (16,126,510))
    fittings : list
        Elements are used to truncate test feature sets.
        Elements should be sklearn.decomposition.pca.PCA or None.
    clf_model : sklearn model
        Used to predict class label of *image*.
    verbose : bool
        If 'True', ranked results are returned as PrettyTable object
    Returns
    -------
    trans : list
        List of class labels & prediction scores, sorted by score.
        First item in list is the top-ranked.
    """

    try:
        morphCoeffs = extraction.morphometric_sample(square)
        chromCoeffs = tuple(extraction.chrom_sample(square))
        gaborCoeffs = list(map(lambda x: extraction.gabor_sample(square,x,3),[8,4,2,1]))
        gaborCoeffs = tuple(np.concatenate(gaborCoeffs))

        sample = tuple(np.concatenate([morphCoeffs,chromCoeffs,gaborCoeffs]))
    except: print ('Error occurred during feature extraction.'); return None

    return id_from_coeffs(sample,featureLengths,fittings,clf_model,verbose=verbose)

def randomSample(l,n):
    """Returns a random, non-repeating sample of length *n* from list *l*

    Parameters
    ----------
    l : list or array
        List to be sampled from.
    n : int
        Length of desired sample. Must be <= length of *l*.

    Returns
    -------
    trans : list
        Random, non-repeating list of *n* elements from list *l*.
    """
    if n<=len(l):
        a = range(len(l))
        random.shuffle(a)
        return [l[i] for i in a[:n]]
    else: raise RuntimeError('n must be <= to length of l. Got %s.' % n)
