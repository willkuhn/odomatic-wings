"""
Train a binary classifier that determines if a pixel is 'wing' or 'background'

Note: Full scratch script with extra analyses located at:
WING SCANING PROJECT/dev/scratch-masking.py

@author: Will
"""
# Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import autoID

# Read config file
cfg = autoID.utils.read_config('config.yaml')

# Import train_df from CSV
train_df = pd.read_csv(cfg['masking_data_path'],header=0,index_col=None)

data = train_df[['Cr','Cb','S','MC6','MC12','E10']]
labels = train_df['type']=='pos' # convert labels to bool (True=pos,False=neg)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('MLP Neural Network classifer')
# MLPC is fastest of all classifiers tried so far, making fairly high-quality masks.
clf = MLPClassifier(activation='relu',solver='lbfgs',alpha=0.1,
                    learning_rate='adaptive',hidden_layer_sizes=(20,20,20),
                    warm_start=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))
clf.fit(X_test,y_test);

from sklearn.externals import joblib
clf_fn =    'MLP xformed pixel classifier.pkl'
scaler_fn = 'StandardScaler for xformd pixels.pkl'
joblib.dump(clf,os.path.join(cfg['mask_model_path'],clf_fn));
joblib.dump(scaler,os.path.join(cfg['mask_model_path'],scaler_fn));

""" NOTES:
  - KNeighborsClassifier scores highly and performs better around edges
    and overall on images, BUT it's 80x slower than KNC
    OPT PARAMS: n_neighbors=10,weights='distance',algorithm='ball_tree'
  - RandomForestClassifier makes similar-quality masks as MLP, but RF is 5x slower
    OPT PARAMS: min_samples_split=2,n_estimators=25,max_depth=None
  - GuassianNB scores similarly to MLP & KNN but quality is bad on real images
    NO PARAMS: GaussianNB()
"""
