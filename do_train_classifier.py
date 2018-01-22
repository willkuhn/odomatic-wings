"""Trains and validates a classifier model using features from
    extracted_features & according to params in config.txt

Example usage:
python do_train_classifier.py -ns
# defaults to save classifier to 'models' folder.
# '-ns' option turns off model saving.
"""

### Setup ======================================================================

import os
from datetime import datetime
import pandas as pd
import autoID
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle
import argparse

def safe_filepath(path,fn):
    """Returns a version of fn that is unused in path to prevent overwriting."""
    f = fn
    while True: # loop until unused filename is found
        if os.path.isfile(os.path.join(path,f)): # if `f` is already in `path`
            b,e = os.path.splitext(f)
            if (b[-2] is '_') & b[-1].isdigit(): # if base of `f` ends with _#
                f = b[:-1]+str(int(b[-1])+1)+e
            else:
                f = b+'_1'+e #append '_1' to base of `f`
        else: break
    return f

# Parse arg: whether to save model
ap = argparse.ArgumentParser()
ap.add_argument('-ns','--no_save', action='store_false', default=True,
	help="Don't save model or transformations")
args = vars(ap.parse_args())
saveModel = args['no_save']


print ('\nSetting up ...')

# Read config file
cfg = autoID.utils.read_config('config.yaml')

# Import metadata
print ('Getting metadata ...')
md = pd.read_csv(cfg['metadata_path'],header=0,index_col=None)

# Import features
print ('Getting features ...')
fd = pd.read_csv(cfg['features_path'],header=0,index_col=0)


### Build training / testing sets ==============================================
print ('Building training & testing sets ...')

# Make `Name` column, where strs in cols specified in 'groupClassBy' are joined
md['Name'] = [' '.join(i) for i in md[cfg['groupClassBy']].values]

# Make `Label` column, where classes with insufficient indivs are relabeled 'rare_unknown'
gb = md.groupby('Name').count()['Species'] # Groupby name & tally each name
rare_classes = gb[gb<cfg['minClassSize']].index # classes with insufficient indivs
md['Label'] = md['Name'] # copy 'Name' column
md.loc[md['Label'].isin(rare_classes),'Label'] = 'rare_unknown' # relabel rares

# Find images for which we have both metadata and features
all_target_fns = list(set(md['img_filename']) & set(fd.index))

# Make 2 helpful dictionaries
fn2label = dict(zip(md['img_filename'],md['Label'])) #translates filename to label
fn2name = dict(zip(md['img_filename'],md['Name'])) #translates filename to name

# Get targets for those common images
all_targets = [fn2label[i] for i in all_target_fns]

# Split all_target_fns into training and testing sets
fns_train,fns_test,target_train,target_test = \
   autoID.utils.stratified_train_test_split(all_target_fns,all_targets,
      train_class_size=cfg['ttf'],
      chop_large_classes=cfg['chop_large_classes'])

# Get training and testing data by looking up features with fns_train & fns_test
data_train = fd.loc[fns_train]
data_test = fd.loc[fns_test]
print (' - Set shapes: training = {}; testing = {}'.format(data_train.shape,data_test.shape))


### Process features ===========================================================
print ('Processing training and testing features ...')
# Here, features are rescaled then ordination is applied with PCA.
# Transformations fit to training data are applied to both training & testing
# data, and saved for later to apply to novel features.

# Set up transformations
ss = StandardScaler()
pca = PCA(n_components=cfg['n_components'],whiten=False)

# Fit & simultaneously transform training data
data_train_t = pca.fit_transform(ss.fit_transform(data_train))

# Transform testing data using fitting from training data
data_test_t = pca.transform(ss.transform(data_test))

# Visualize data before/after transformation
#plt.matshow(data_train[:50]) # before transformation
#plt.matshow(ss.transform(data_train)[:50]) # with rescaling transformation
#plt.matshow(data_train_t[:50]) # with rescaling + PCA

# Visualize PC loadings and explained variance of PCs
#plt.plot(pca.components_[0],'r',pca.components_[1],'b',alpha=0.5)
#plt.plot(pca.explained_variance_ratio_)

print (' - Set shapes: training = {}; testing = {}'.format(data_train_t.shape,data_test_t.shape))


### Train / validate classifier ================================================

print ('Training classifier ...')
# Set up classifier
clf = autoID.classifier.setup_classifier(cfg['clfMethod']) # set up classifier

# Train classifier
t0 = datetime.now()
clf.fit(data_train_t,target_train)
fit_time = datetime.now()-t0
print (' - Done. Trained on {} samples in {} h:m:s'.format(len(data_train_t),fit_time))

# Validate classifier
print ('Validating classifier ...')
t0 = datetime.now()
predicted = clf.predict(data_test_t)
pred_time = datetime.now()-t0
print (' - Done. Predicted {} samples in {} h:m:s'.format(len(data_test_t),pred_time))


### Evaluate classifier ========================================================

print (' - \n\nClassifier description:\n{}\n\n'.format(clf))

# Output per-class report and overall accuracy
print ('\n - Classification report:\n{}\n'.format(
        metrics.classification_report(target_test, predicted)))

print (' - Number of classes: %d' % len(clf.classes_))

# Top-1 accuracy
acc = metrics.accuracy_score(target_test, predicted)*100
print (' - Top-1 accuracy: {:.2f}%'.format(acc))

if hasattr(clf,'predict_proba'): # report top-k-accuracies
    probs = clf.predict_proba(data_test_t)
    classes = clf.classes_
    top3,top5,top10 = [autoID.utils.top_k_accuracy_score(\
                       target_test,probs,classes,k=i)*100 for i in [3,5,10]]
    print (' - Top-3 accuracy: {:.2f}%'.format(top3))
    print (' - Top-5 accuracy: {:.2f}%'.format(top5))
    print (' - Top-10 accuracy: {:.2f}%'.format(top10))
    print ('\n')

    # print ('\n - Top 5 classification report:\n{}\n'.format(
    #    autoID.utils.top_k_classification_report(target_test, probs, classes,5))

# Output confusion matrix
autoID.utils.confusion_plot(target_test,predicted)


### Save model & feature transformation ========================================
if saveModel:
    print ('Saving transformations & model ....')


    # Give model file a safe filename
    modelPath = os.path.join(os.getcwd(),cfg['model_path'])
    model_fn = '{}_{}cl_{:.0f}ac.pkl'.format(cfg['clfMethod'],len(clf.classes_),acc)
    model_fn = safe_filepath(modelPath,model_fn)

    # Save model + transformation
    modelPacket = {'scaler':ss,'pca':pca,'model':clf,
             'params':{'groupClassBy':cfg['groupClassBy'],
                       'minClassSize':cfg['minClassSize'],
                       'extraClass':cfg['extraClass'],
                       'ttf':cfg['ttf'],
                       'chop_large_classes':cfg['chop_large_classes'],
                       'n_components':cfg['n_components'],
                       'clfMethod':cfg['clfMethod']}
        }
    with open(os.path.join(modelPath,model_fn),'wb') as f:
        pickle.dump(modelPacket,f,protocol=pickle.HIGHEST_PROTOCOL)
        #joblib.dump(modelPacket,f,compress=3,protocol=-1)
    print (' - Done. Pickled to {}'.format(os.path.join(modelPath,model_fn)))
else:
    print ('Model not saved.')
