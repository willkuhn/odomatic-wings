# -*- coding: utf-8 -*-
"""
AUTOMATIC IDENTIFICATION FRAMEWORK

-This version updated after my dissertation.-
@author: Will
"""
import os
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
import matplotlib.patches as patches
import time
from sklearn.pipeline import Pipeline#, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler#normalize
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
import pandas as pd
import itertools
import random
import pickle #import cPickle as pickle
import skimage

import autoID
from autoID.modelBuilding import truncate_train_features,truncate_test_features

"""
===========================================
Parameters for configuring autoID framework
===========================================
"""
#Training & testing sets:
groupClassBy = ['Genus','Species']#,'Sex'] #list : how to group individuals
ttf = 0.75 #float (0 to 1) : training:testing ratio
pcCutoff = 50#20#'mle' #int : Max n_features in each feature set
minClassSize = 15 #int : cutoff for reassigning specimens to 'rare_unknown'
extraClass = True #bool : Whether 'rare_unknown' should be included as a class

#Features preprocessing & classifier model setup:
featureLengths = (15,138,510) #tuple : n_coeffs in each features set
clfMethod = 'rf'#'svm_rbf_gs' #str ('rf','nb',svm_lin','svm_other','nn) : classification algorithm
#classifierType = 'OneForAll' #str : ('OneForAll','OneForEach')
#ofeMethod = 'random' #str : Method used to build zero-sets in a OneForEach
#-type classifier
#checkClasses = 'True' #bool : Classes that are indistinguishable should be
#lumped together (based on their Mahalanobis distance)

classifier_methods = \
{'rf': CalibratedClassifierCV(RandomForestClassifier(oob_score='True',max_features=0.1,n_estimators=90),method='sigmoid'),
'nb': CalibratedClassifierCV(GaussianNB(),method='sigmoid'),#Can't be optimized
'svm_rbf_g': SVC(kernel='rbf',C=1000000.0, gamma=1.0000000000000001e-05,probability=True), #RBF-SVM, optimized for genus, sigmoidal prob calibration is built-in, I think
'svm_rbf_gs': SVC(kernel='rbf',C=100.0, gamma=0.10000000000000001,probability=True), #RBF-SVM, optimized for genus-species, sigmoidal prob calibration is built-in, I think
'svm_rbf_gss': SVC(kernel='rbf',C=100.0, gamma=0.10000000000000001,probability=True), #RBF-SVM, optimized for genus-species-sex
'svm_lin': SVC(kernel='linear',probability='True',C=10.0,gamma=1.0), #Linear SVM, optimized
'svm_poly':SVC(kernel='poly',probability='True',C=1.0,gamma=1.0),#Optimized, but not working "integer is required"
'nn': CalibratedClassifierCV(Pipeline(steps=[('rbm', BernoulliRBM(n_components=256, \
    verbose=False,batch_size=30,learning_rate=0.01,n_iter=5)), ('logistic', \
    linear_model.LogisticRegression(C=100000.0))]),method='sigmoid'), #NN, optimized
'ada': AdaBoostClassifier(SVC(probability=True,kernel='linear',C=10.0,gamma=1.0),\
    n_estimators=50,learning_rate=1.0,algorithm='SAMME') #Ada part needs optimization
}
#def pdet(arr):#pseudo determinant: product of all non-negative eigenvalues
#    #See: https://en.wikipedia.org/wiki/Determinant
#    e = [i for i in np.linalg.eigvals(arr) if i>=0.]
#    return np.product(e)

def mahalanobisDist(array1,array2):#See: https://en.wikipedia.org/wiki/Bhattacharyya_distance
    m1 = np.mean(array1,axis=0)
    cov1 = np.cov(array1,rowvar=0)
    m2 = np.mean(array2,axis=0)
    cov2 = np.cov(array2,rowvar=0)
    cov = (cov1+cov2)/2
    m = m1-m2
    mT = np.array([[i] for i in m])#Transpose m
    d = np.sqrt(np.dot(np.dot(m,np.linalg.inv(cov)),mT)[0])#/len(m)**2
#    print (d)
    return d

def make_Target2(md,target_fns,minClassSize):
    """Adds a column 'Target2' to `md`: a copy of md.Target, but with rares
    labeled as 'rare_unknown' based on minClassSize"""
    t = md[md.img_filename.isin(target_fns)].Target
    t1 = t.groupby(t).count()
    rares = t1[t1<minClassSize].index
    md['Target2'] = map(lambda x: 'rare_unknown' if x in rares else x,md.Target.values)
#%% SET DIRECTORIES, GET FILENAMES=============================================
filenames = [i for i in os.listdir(image_path) if i.endswith('.tif')] #List of file names in directory
filenames_in_squares_path = [i for i in os.listdir(squares_path) if i.endswith('.tif')]
new_filenames = [fn for fn in filenames if fn not in filenames_in_squares_path]
print ('New files: '+str(len(new_filenames)))
#%% UNLOOPED IMAGE STANDARDIZATION=============================================
#Get image file:
fn = random.choice(filenames) #Get random filename
img = nd.imread(os.path.join(image_path,fn)) #Read image file
square = autoID.preprocessing.make_square(img) #Standardize image
print (fn)
if square is not None:
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(square)
    ax2.set_title('Standardized image')
    plt.tight_layout()
    plt.figtext(0.5,0.965,fn,ha='center',color='black',weight='bold',size='large')
    plt.show()
#skimage.io.imsave(os.path.join(squares_path,fn), square)
#%% LOOPED IMAGE PREPROCESSING=================================================
#Turn (new) scans into squares:
for fn in new_filenames:#filenames:
    try:
        img = nd.imread(os.path.join(image_path,fn))
        square = autoID.preprocessing.make_square(img)
        skimage.io.imsave(os.path.join(squares_path,fn),square)
    except:
        print ('Preprocessing error: %s' % (fn))
        continue
#%% FIX SOME SPECIFIC SQUARES (LEFTIES & SWITCHIES)============================
#**Requires `id2filename` below.
#Reflect left wings to match the right ones:
lefties = [i for i,j in zip(specIDs,md['Condition'].values) if 'left' in j.lower()] #Find lefties based on text in Condition column
leftiesFns = [id2filename[i] for i in lefties if id2filename[i] in os.listdir(squares_path) and id2filename[i] in new_filenames]
for i in leftiesFns:
    print (i)
    imgl = nd.imread(os.path.join(squares_path,i))
    skimage.io.imsave(os.path.join(squares_path,i), np.fliplr(imgl))#Reflect
    del imgl

#f=[]
#for n in ['895','1584','2509','2486']:
#    f.append([i for i in filenames if n in i][0])

#Switch fore- and hingwings for some specimens:
switchies = [i for i,j in zip(specIDs,md['Condition'].values) if 'switch' in j.lower()] #Find switchies based on text in Condition column
switchiesFns = [id2filename[i] for i in switchies if id2filename[i] in os.listdir(squares_path) and id2filename[i] in new_filenames]
for i in switchiesFns:
    print (i)
    imgsw = nd.imread(os.path.join(squares_path,i))
    length=len(imgsw)
    imgsw = np.concatenate((imgsw[length//2:],imgsw[:length//2]))#Wing switch
    skimage.io.imsave(os.path.join(squares_path,i), imgsw)
    del imgsw
#%% LOOPED EXTRACT DATA FROM SQUARES===========================================
#Extract data from squares: (For each filename (key) in filename2label, if
#filename exists in squares_path, import image and extract data)
t0=time.clock()
#dataDict = {}
dataDF = pd.DataFrame()

square_filelist = [i for i in os.listdir(squares_path) if i.endswith('.tif')]
for fn in square_filelist:
    #if not dataDict.has_key(fn):
    if fn not in dataDF.index:
#for fn in filename2label.keys():
#    if (fn in square_filelist and not dataDict.has_key(fn)): #if filename is in squares folder and is not already in dict
        try:
            coeffs = []
            square = nd.imread(os.path.join(squares_path,fn))
            morphCoeffs = autoID.extraction.morphometric_sample(square)
            chromCoeffs = autoID.extraction.chrom_sample(square)
            gaborCoeffs = map(lambda x: autoID.extraction.gabor_sample(square,x,3),[8,4,2,1])
            gaborCoeffs = np.concatenate(gaborCoeffs)
            coeffs.extend(list(morphCoeffs))
            coeffs.extend(list(chromCoeffs))
            coeffs.extend(list(gaborCoeffs))
            #dataDict[fn] = coeffs
            dataDF = dataDF.append(pd.DataFrame(dict(zip(range(len(coeffs)),coeffs)),index=[fn]))
        except:
            print ('Extraction failure: %s' % (fn))
            continue#dataDict[fn] = None
t1=time.clock()
print ('Time to complete: %f s' % (t1-t0))
print ('Average time per specimen: %f s' % ((t1-t0)/len(dataDF)))
#del square_filelist,square,morphCoeffs,chromCoeffs,gaborCoeffs
#Time to complete: 19855.934378 s
#Average time per specimen: 11.418019 s
#%% EXPORT/IMPORT PRE-EXTRACTED FEATURES=======================================
#dataDF.to_csv('extractedData.csv') #Export
#pickle.dump(dataDict,open('extractedData.pkl','wb'),-1) #Export
dataDF = pd.read_csv('extractedData.csv',index_col=0) #Import
dataDF.columns = dataDF.columns.astype(int) #Convert col labels to int type
#dataDict=pickle.load(open('extractedData.pkl','rb')) #Import
print ('Database shape: ', dataDF.shape)
#print ('Total specimens in database: %d' % len([i for i in dataDict.keys()]))

#dataDF = pd.DataFrame.from_dict(dataDict,orient='index')
#%%
#Get specimen metadata from spreadsheet:
md = pd.read_excel('D:\Dropbox\Rutgers\Research\Wing scans\Wing Scans Specimen List.xlsx')
#Curate data:
md.Genus = md['Genus'].str.title()
md.Species = md['Species'].str.lower()
md.Sex = md['Sex'].str.upper()
md.Suborder = md['Suborder'].str.title()
#Quality control specimens:
md = md[md['Suborder'].isin(['Anisoptera','Zygoptera'])]
md = md[md['Sex'].isin(['F','M'])]
md = md[~(md['Species'].str.contains('cf|\.|\?|spp|unk',na=True)|(md['Species'].isin(['sp',' '])))]
md.Species = md['Species'].apply(lambda x: x.split(' ')[0])#Remove subspecies, keep only specific epithet
#Make 'numCode' col: #### from WRK-WS-#### in 'Scan ID' col:
md['numCode'] = map(autoID.modelBuilding._str2num,md['Scan ID'].values)
#Add 'img_filename' col: image filenames for rows by matching numCodes:
imgNumCode2filename = dict(zip( map(autoID.modelBuilding._str2num,filenames), filenames))
def get_fn(numCode): #Catch numCodes in metadata file but not image filenames
    try: return imgNumCode2filename[numCode]
    except KeyError: return None
md['img_filename'] = map(get_fn,md.numCode.values)
#Make 'Target' column, where strs in cols specified in 'groupClassBy' are joined:
md['Target'] = [' '.join(i) for i in md[groupClassBy].values]
del imgNumCode2filename
#pickle.dump(md,open('metadata.pkl','wb'),-1) #Export"""
#%% Export list of specimens:
codes = md[['RU Collection #','Code','Other code','Locality code']]
def combine_codes(row):
    row = row.dropna()
    if len(row)==0: return ''
    else: return '; '.join(row)
codes=codes.apply(combine_codes,axis=1)
md['Codes'] = codes
md['Species Name'] = md[['Genus','Species']].apply(lambda r:' '.join(r),axis=1)
md.sort(columns=['Suborder','Family','Species Name','Sex'])\
.to_excel('D:\Dropbox\Rutgers\Research\Dissertation\Ch3 - AutoID\Appendix A table of specimens.xls',
            columns=['Scan ID','Suborder','Family','Species Name','Sex',
            'Collection Date','Country','State/Province','County',
            'Collection Locality','GPS Coordinates','Collector',
            'Codes'],index=False)
#%%
"""
#Make 'Target2' col: relabel 'Target' vals to 'rare_unknown' if len(class)<minClassSize:
is_rare = dict(md.groupby('Target').Target.count() < minClassSize)
md['Target2'] = map(lambda x: x if is_rare[x] else 'rare_unknown',md.Target.values)
del is_rare"""
"""#%% BUILD CLASS DICTIONARY=====================================================
#Defines classes based on groupClassBy and data spreadsheet:
#Make dict of specimen_IDs : filenames:
imgNumCode2filename = dict(zip( map(autoID.modelBuilding._str2num,filenames), filenames))

#Get specimen metadata from spreadsheet:
md = pd.read_excel('D:\Dropbox\Rutgers\Research\Wing scans\Wing Scans Specimen List.xlsx')
#Curate data:
md.Genus = md.Genus.str.title()
md.Species = md.Species.str.lower()
md.Sex = md.Sex.str.upper()
md.Suborder = md.Suborder.str.title()
#Quality control specimens:
md = md[md.Suborder.isin(['Anisoptera','Zygoptera'])]
md = md[md.Sex.isin(['F','M'])]
md = md[~(md.Species.str.contains('cf|\.|\?|spp|unk',na=True)|(md.Species.isin(['sp',' '])))]
md.Species = md.Species.apply(lambda x: x.split(' ')[0])#Remove subspecies, keep only specific epithet
#Make dict of specimen_IDs : class_labels:
classNames = [' '.join(i) for i in md[groupClassBy].values ] #Combine values from groupClassBy into str for class name
specIDs = [autoID.modelBuilding._str2num(i) for i in md['Scan ID'].values] #Convert specimens IDs 'WRK-WS-#####' to ints from the ##### part of ID
id2label = dict(zip(specIDs,classNames))#Dict of specimen_IDs:class_labels
#Make dict of filenames : class_labels:
filename2label = {} #Dict of filenames:class_labels
for i in imgNumCode2filename:
    try:
        filename2label[imgNumCode2filename[i]] = id2label[i]
    except KeyError:
        filename2label[imgNumCode2filename[i]] = None
#cnt=0
for key, val in filename2label.items(): #Throw out specimens with None for ID
    if val is None: del filename2label[key]#;cnt+=1
del key,val,i,classNames,imgNumCode2filename"""
"""#%% RELABEL CLASSES============================================================
#Reassign small classes to 'rare_unknown' class, based on minClassSize
filename2relabel = filename2label.copy()#Copy of filename2label
counts={}
for key,items in itertools.groupby(sorted(filename2relabel.values())):
    i=0
    for subitem in items:
        i+=1
    counts[subitem]=i
rares = [i for i in counts if counts[i] < minClassSize]
for i in filename2relabel: #Relabel items listOfOthers as 'rare_unknown'
    if filename2relabel[i] in rares:
        filename2relabel[i]=u'rare_unknown'
#If extraClass is False, delete all individuals marked rare_unknown
if not extraClass: #Delete those labeled 'rare_unknown'
    temp = []
    for i in filename2relabel:
        if filename2relabel[i] == 'rare_unknown':
            temp.append(i)
    for i in temp: del filename2relabel[i]
    del temp

#pickle.dump(filename2relabel,open('filename2relabel25.pkl','wb'),-1) #Export"""
"""target_fns = [i for i in dataDict.keys() if i in filename2relabel.keys()]#Filter out unlabeled & extraction-error specimens
data = np.array([dataDict[i] for i in target_fns])#data
target = np.array([filename2relabel[i] for i in target_fns])#target
data_train,data_test,target_train,target_test = \
train_test_split(data,target,train_size=0.75,
                 random_state=np.random.randint(0,1000),stratify=target)
"""
#%% COEFFICIENT ORDINATION/NORMALIZATION=======================================
#Get data and target (dataDict data filtered with filename2relabel):
target_fns = list(dataDF[dataDF.index.isin(md.img_filename.values)].index)
make_Target2(md,target_fns,minClassSize) #Make relabeled md['Target2']

data = np.array([dataDF.loc[i] for i in target_fns])
target = np.array([md[md.img_filename==i].Target2.values[0] for i in target_fns])
t=pd.Series(target); print (t.groupby(t).count());del (t)#printout of class sizes

data_train,data_test,target_train,target_test = \
                    autoID.utils.stratified_train_test_split(data,target,10,5)

#Train scaling on data_train and apply to both sets (makes coeffs mean=0,var=1):
ssc = StandardScaler()
data_train = ssc.fit_transform(data_train)
data_test = ssc.transform(data_test)

#Transform (ordinate) feature sets in training data, individually:
fl = list(np.cumsum(featureLengths)) #Accumulating list of feature lengths

data_train_sets = np.hsplit(data_train,fl[:-1]) #Split into feature sets
f = lambda x: truncate_train_features(x,k=pcCutoff)
data_train_sets_trans,fittings = np.transpose(map(f,data_train_sets)) #Truncate sets
data_train_trans = np.hstack(data_train_sets_trans) #Merge sets

data_test_sets = np.hsplit(data_test,fl[:-1]) #Split into feature sets
f = lambda x,y: truncate_test_features(x,y)
data_test_sets_trans = map(f,data_test_sets,fittings) #Truncate sets
data_test_trans = np.hstack(data_test_sets_trans) #Merge sets

del data_train_sets_trans,data_test_sets_trans,f

'''#Plot data matrices before/after truncation & train-test split
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(10,18))
ax1.imshow(data)#Original data
ax1.set_title('Original Data')
ax2.imshow(data_train_trans)#Training data
ax2.set_title('Training Data')
ax3.imshow(data_test_trans)#Testing data
ax3.set_title('Testing Data')
'''
#%% TRAIN & VALIDATE CLASSIFIER MODEL==========================================
t0=time.clock()
clf= classifier_methods[clfMethod]
clf_model = clf.fit(data_train_trans,target_train)

predicted = clf_model.predict(data_test_trans)
t1=time.clock()
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(target_test, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(target_test, predicted))
print ('Time to complete: %f s' % (t1-t0))
print ('Number of classes: %d' % len(clf_model.classes_))
#%% EXPORT/IMPORT CLASSIFIER MODEL & FITTINGS==================================
#pickle.dump((clf_model,fittings),open('classifierModel_spp_sex.pkl','wb'),-1) #Export
#clf_model,fittings = pickle.load(open('classifierModel1.pkl','rb')) #Import
#%% ID A RANDOM SPECIMEN FROM IMAGE OR SQUARE==================================

#Get image file:
fn = list(set(dataDict.keys()) & set(filename2relabel.keys()))
fn = fn[np.random.randint(0,len(fn)-1)] #Get random filename
t0=time.clock()
img = nd.imread(os.path.join(image_path,fn)) #Read image file
square = autoID.make_square(img) #Preprocess image
if filename2relabel[fn]!=filename2label[fn]:
    print ('True ID: %s (%s)' % (filename2relabel[fn],filename2label[fn]))
else: print ('True ID: %s' % filename2relabel[fn])
print (fn)
#autoID.modelBuilding.id_from_image(img,featureLengths,fittings,clf_model,verbose='True')
autoID.modelBuilding.id_from_square(square,featureLengths,fittings,clf_model,verbose='True')
t1=time.clock()
#print ('Time to complete: %f s' % (t1-t0))
#%% ID A RANDOM SPECIMEN FROM dataDict COEFFICIENTS============================
fn = list(set(dataDict.keys()) & set(filename2relabel.keys()))
fn = fn[np.random.randint(0,len(fn)-1)]
if filename2relabel[fn]!=filename2label[fn]:
    print ('True ID: %s (%s)' % (filename2relabel[fn],filename2label[fn]))
else: print ('True ID: %s' % filename2relabel[fn])
print (fn)
t0=time.clock()
autoID.modelBuilding.id_from_coeffs(dataDict[fn],featureLengths,fittings,clf_model,verbose='True')
t1=time.clock()
#print ('Time to complete: %f s' % (t1-t0))
#%% GET EXAMPLAR SQUARES FOR PICTORIAL RESULTS, BELOW:
classes = list(clf_model.classes_) #list of classes in trained model
square_filelist = os.listdir(squares_path)
squares = {}
for cl in classes:
    files = [i for i in filename2relabel.keys() if filename2relabel[i] == cl and i in square_filelist]
    if len(files) > 50: files=np.random.choice(files,50)#Limit size of large classes

    if cl == 'rare_unknown': #Find mean image for rare_unknown
        mean_square = autoID.utils.meanImage(files,dir=squares_path,func=np.mean)
    else: #Find median image for all others
        mean_square = autoID.utils.meanImage(files,dir=squares_path,func=np.median)

    squares[cl] = mean_square
del cl,i,files,mean_square
#%%
class_examples_path = 'D:\Dropbox\Rutgers\DSA 2015\Class Examples'
for k,v in squares.items():
    skimage.io.imsave(os.path.join(class_examples_path,k+'.tif'),v)
#%% DEMO: ID A RANDOM SPECIMEN FROM IMAGE: PICTORAL RESULTS====================
import textwrap
fnlist = list(set(dataDict.keys()) & set(filename2relabel.keys()))#set(i for i in filename2relabel if filename2relabel[i] != 'rare_unknown'))

#for fn in os.listdir(fig_path):
#for i in range(2):
#GET IMAGE & DO IDENTIFICATION:
t0=time.clock()
fn = np.random.choice(fnlist) #Get random filename
fn=fn.split('.')[0]+'.tif'
img = nd.imread(os.path.join(image_path,fn)) #Read image file
square = autoID.make_square(img) #Preprocess image
bb=map(autoID.preprocessing.bounding_box,autoID.preprocessing.sort_masks(autoID.preprocessing.get_image_mask(img)))
results = autoID.modelBuilding.id_from_square(square,featureLengths,fittings,clf_model,verbose='False')
t1=time.clock()

if filename2relabel[fn]!=filename2label[fn]:
    print ('True ID: %s (%s)' % (filename2relabel[fn],filename2label[fn]))
else: print ('True ID: %s' % filename2relabel[fn])
print ('Time to complete: %f s' % (t1-t0))

#Red flag this ID?
if results[0][1]-results[1][1] < 0.35: flag = 'True'
else: flag = 'False'

#MAKE FIGURE:
fig = plt.figure(figsize=(10,8))

grid = gridspec.GridSpec(2,2,wspace=0.05,height_ratios = [2.1,1])

#Original scan w/bboxes:
ax = plt.Subplot(fig,grid[0])
ax.imshow(img)
ax.set_xticks([]);ax.set_yticks([])
for b in bb: #for each bounding box
    verts = [
        (b[1],b[2]), # left, bottom
        (b[1],b[0]), # left, top
        (b[3],b[0]), # right, top
        (b[3],b[2]), # right, bottom
        (0., 0.), # ignored
        ]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1,ec='r')
    ax.add_patch(patch)
    ax.annotate('Input image',
        xy=(2,2),  #dummy coords
        xytext=(0.05, 0.05), # fraction, fraction (from bottom left)
        textcoords='axes fraction',
        color = 'b',weight='bold'
        )
fig.add_subplot(ax)

#Square:
ax = plt.Subplot(fig,grid[1])
ax.imshow(square)
ax.set_xticks([]);ax.set_yticks([])
ax.annotate('Preprocessed image',
    xy=(2,2), #dummy coords
    xytext=(0.05, 0.05), #fraction, fraction
    textcoords='axes fraction',
    color = 'b',weight='bold'
    )
fig.add_subplot(ax)

#Results squares:
n_results = 4 #How many results to display
res = gridspec.GridSpecFromSubplotSpec(1,n_results,subplot_spec=grid[1,:],wspace=0.0,hspace=0.0)

for k,(i,j) in enumerate(zip(res,results[:n_results])):#For each result
    #Change color of 1st result annotations based on flag:
    if k==0 and flag=='True': c='r'
    elif k==0 and flag=='False': c='g'
    else: c='b'

    ax = plt.Subplot(fig,i)
    if k==0: ax.set_title('Best matches:')
    ax.imshow(squares[j[0]])
    ax.axis('off')
    ax.annotate(textwrap.fill(j[0],17),#Annotate with class name
            xy=(2,2), #dummy coords
            xytext=(0.05, 0.53), #fraction, fraction
            textcoords='axes fraction',
            color=c,weight='bold'
            )
    ax.annotate('Score: %1.4f' % j[1],#Annotate with class score
            xy=(2,2), #dummy coords
            xytext=(0.05, 0.05), #fraction, fraction
            textcoords='axes fraction',
            color=c,weight='bold'
            )
    fig.add_subplot(ax)

if filename2relabel[fn]!=filename2label[fn]:
    plt.suptitle('True ID: %s (%s)' % (filename2relabel[fn],filename2label[fn]))
else: plt.suptitle('True ID: %s' % filename2relabel[fn])
#    fig_path=r'D:\Dropbox\Rutgers\Research\Dissertation\Ch3 - AutoID\Python framework scripts\result examples'
#    fn0=fn.split('.')[0]+'.png'
#    plt.savefig(os.path.join(fig_path,fn0))
plt.show()
#%% DEMO: ID A RANDOM SPECIMEN FROM IMAGE: TABULAR RESULTS=====================
from matplotlib.path import Path
import matplotlib.patches as patches
#Get image & perform ID:
t0=time.clock()
fn = list(set(dataDict.keys()) & set(filename2label.keys()))
fn = fn[np.random.randint(0,len(fn)-1)] #Get random filename
img = nd.imread(os.path.join(image_path,fn)) #Read image file
square = autoID.make_square(img) #Preprocess image
bb=map(autoID.preprocessing.bounding_box,autoID.preprocessing.sort_masks(autoID.preprocessing.get_image_mask(img)))
results = autoID.modelBuilding.id_from_square(square,featureLengths,fittings,clf_model,verbose='False')
t1=time.clock()

print ('\nFilename: %s\nLabel: %s' % (fn, filename2label[fn]))
print ('Time to complete: %f s' % (t1-t0))

#Red flag this ID?
if results[0][1]-results[1][1] < 0.35: flag = 'True'
else: flag = 'False'

fig = plt.figure()
#fig.set_size_inces(10,20)

#Original image:
ax1= fig.add_subplot(2,2,1)
ax1.imshow(img)
ax1.set_title('Original image')
ax1.axis('off')
for b in bb: #for each bounding box
    verts = [
        (b[1],b[2]), # left, bottom
        (b[1],b[0]), # left, top
        (b[3],b[0]), # right, top
        (b[3],b[2]), # right, bottom
        (0., 0.), # ignored
        ]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1,ec='b')
    ax1.add_patch(patch)

#Preprocessed image:
ax2 = fig.add_subplot(2,2,3)
ax2.imshow(square)
ax2.set_title('Preprocessed image')
ax2.axis('off')

#ID table:
columns = [groupClassBy[-1],'Score']
cell_text = []
for row in results[:10]:
    cell_text.append([row[0],round(row[1],4)])
ax3 = fig.add_subplot(1,2,2)
tble = plt.table(cellText=cell_text,colLabels=columns,loc='center',fontsize=40,colWidths = [0.75, 0.25])
ax3.add_table(tble)
ax3.axis('off')
ax3.set_title('Label: %s' % filename2label[fn])
tble.scale(1.2,1.2)

#fig.tight_layout()
plt.show()
#%% STATS ON WING SCANS DATASET===============================================
c = counts.values()
c = np.sort(c)
print ('Total specimens: %d' % np.sum(c))
print ('Specimens grouped by %s' % str(groupClassBy))
print ('Total classes: %d' % len(c))
print ('Classes with 2+ individuals: %d' % len(c[c>=2]))
print ('Classes with 5+ individuals: %d' % len(c[c>=5]))
print ('Classes with 8+ individuals: %d' % len(c[c>=8]))
print ('Classes with 10+ individuals: %d' % len(c[c>=10]))
print ('Classes with 20+ individuals: %d' % len(c[c>=20]))
n, bins, patches = plt.hist(c, 20, facecolor='green', alpha=0.75)
plt.xlabel('Class size')
plt.ylabel('Frequency')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()
#%% DETERMINE RED FLAGGING CUTOFF==============================================
predicted = clf_model.predict_proba(data_test_trans)#class probabilities for each test specimen
classes=clf_model.classes_ #list of classes
truePositives = []
falsePositives = []
for pred,true in zip(predicted,target_test):
    order = np.argsort(pred)[::-1]
    if classes[order[0]] == true:
        truePositives.append(pred[order])
    else: falsePositives.append(pred[order])

#True positives:
fig,(ax1,ax2) = plt.subplots(1,2)
plt.subplot(141)
plt.plot(np.transpose(truePositives),'k-')
plt.title('True Positives')
plt.xlim(0.,10.)
plt.ylim(0.,1.)

#False positives:
plt.subplot(142)
plt.plot(np.transpose(falsePositives),'k-')
plt.title('False Positives')
plt.xlim(0.,10.)
plt.ylim(0.,1.)

#Diff of 1st and 2nd predictions:
tpDiff = [i[0]-i[1] for i in truePositives]
fpDiff = [i[0]-i[1] for i in falsePositives]
plt.subplot(143)
plt.boxplot([tpDiff,fpDiff],whis=[5,95],showmeans=True,labels=['TP[0]-TP[1]','FP[0]-FP[1]'])

#First predictions only:
plt.subplot(144)
plt.boxplot([[i[0] for i in truePositives],[i[0] for i in falsePositives]],whis=[5,95],showmeans=True,labels=['TP[0]','FP[0]'])
plt.show()
#Cutoff of 0.35 or lower looks good
#%%
import profile
profile.run('autoID.morphometric_sample(square)')
profile.run('autoID.extraction.chrom_sample(square)')
profile.run('autoID.extraction.gabor_sample(square,8,3)')
#%% TUNING CLASSIFIER HYPERPARAMETERS==========================================
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
X_train=data_train_trans;y_train=target_train;X_test=data_test_trans;y_test=target_test

'''Set classifier & parameter grid here:'''
#cl = RandomForestClassifier()
#params = [{'n_estimators': [70,80,90,100], 'max_features': [0.25,0.5,0.75,None,'sqrt','log2'],'oob_score':['True','False']}]
#best params: {'max_features': '0.1', 'n_estimators': 90, 'oob_score': 'True'}

#cl = AdaBoostClassifier()
#params = [{'n_estimators': [30,40,50,60,70], 'learning_rate': [0.1,0.2,0.3]}]
#best params: ?

cl=SVC(kernel='rbf') #RBF-SVM
C_range = np.logspace(-3,3,1)
gamma_range = np.logspace(-3,3,1)
params = [dict(gamma=gamma_range, C=C_range)]
#best params: {'C': 100.0, 'gamma': 0.10000000000000001} for genus-species-sex
#             {'C': 10.0, 'gamma': 1.0} gives results very near to above for g-s-s
#             {'C': 100.0, 'gamma': 0.10000000000000001} for genus-species
#             {'C': 1000000.0, 'gamma': 1.0000000000000001e-05} for genus

#cl=SVC(kernel='linear') #Linear SVM
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#params = [dict(gamma=gamma_range, C=C_range)]
#best params: {'C': 10.0, 'gamma': 1.0}

#cl=SVC(kernel='poly') #Linear SVM
#C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)
#params = [dict(gamma=gamma_range, C=C_range)]
#best params: {'C': 1.0, 'gamma': 1.0}

#cl = Pipeline(steps=[('rbm', BernoulliRBM()), ('log', linear_model.LogisticRegression())]) #NN
#params = {'rbm__learning_rate':[0.05,0.01],'rbm__n_iter':[5,10,15,20],'rbm__n_components':2**np.arange(3,10),'rbm__batch_size':[5,10,15,20,25,30],'log__C':np.logspace(-2, 10, 13)}
#best params: {'log__C': 1000000.0, 'rbm__batch_size': 30, 'rbm__learning_rate': 0.01, 'rbm__n_components': 256, 'rbm__n_iter': 5}


'''Run this for grid search:'''
#scores = ['precision', 'recall']
scores = ['precision']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(cl, param_grid=params, cv=3,scoring='%s_weighted' % score,error_score=0,n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#%%
from sklearn.feature_selection import f_classif
fs,pvals=f_classif(data_train_trans,target_train)
#%% VIEW SPECIES IN POINT CLOUD (PCA)==========================================
from sklearn.decomposition import PCA
from sklearn.lda import LDA

#Truncate data feature:
data_sets = np.hsplit(data,fl[:-1]) #Split into feature sets
data_sets_trans,fittings = np.transpose(map(lambda x: autoID.modelBuilding.truncate_train_features(x,k=pcCutoff),data_sets)) #Truncate sets
data_trans = np.hstack(data_sets_trans) #Merge sets

classNames = [' '.join(i) for i in md[['Genus','Species','Sex']] ] #Combine values from groupClassBy into str for class name
specIDs = [autoID.modelBuilding._str2num(i) for i in md['Scan ID'].values] #Convert specimens IDs 'WRK-WS-#####' to ints from the ##### part of ID
id2label = dict(zip(specIDs,classNames))#Dict of specimen_IDs:class_labels
filename2label = {} #Dict of filenames:class_labels
for i in id2filename:
    try:
        filename2label[id2filename[i]] = id2label[i]
    except KeyError:
        filename2label[id2filename[i]] = None
for key, val in filename2label.items(): #Throw out specimens with None for ID
    if val is None: del filename2label[key]
del key,val,i,specIDs,classNames,id2label

#Do PCA:
pca=PCA(n_components=None,whiten=False)
pca.fit(data_trans)
var = pca.explained_variance_ratio_
pcaT=pca.transform(data_trans)

#Do LDA:
lda = LDA(n_components=None)
ldaT = lda.fit(data_trans,target).transform(data_trans)

X = pcaT#ldaT

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6))
yy = target
classes=np.unique(yy)[:-1]#Leave out 'unknown' class
color=iter(plt.cm.nipy_spectral(np.linspace(0,1,len(classes))))
for cl in classes:
    c=next(color)
    m=list(np.mean(X[yy==cl],axis=0)[:2])#Mean location
    #Plot group rays:
    for x,y in X[yy==cl][:,:2]:
        ax1.plot((x,m[0]),(y,m[1]),'-',c=c,alpha=0.3)
    #Plot points:
    ax1.plot(X[yy==cl,0], X[yy==cl,1], 'o', alpha=0.3,
    c=c, label=cl)
    #Annotate with group label:
    ax1.annotate(cl,#Annotate with class name
            xy=(0,0),xycoords='data', #dummy coords
            xytext=m,textcoords='data',
            color='k',fontsize=8#c#,weight='bold'##,
            )
    #Print generalized & total variance for class:
    print (cl,': \t', np.linalg.det(np.cov(X[yy==cl].T)), '\t',np.trace(np.cov(X[yy==cl].T)))
ax1.set_xlabel('PC 1 (%.2f %% of total variance)' % (var[0]*100))#,weight='bold',fontsize='xx-large');
ax1.set_ylabel('PC 2 (%.2f %% of total variance)' % (var[1]*100))#,weight='bold',fontsize='xx-large');
#ax1.set_xlabel('LD 1')#,weight='bold',fontsize='xx-large');
#ax1.set_ylabel('LD 2')#,weight='bold',fontsize='xx-large');


#Distance matrix:
dists = np.zeros((len(classes),len(classes)))
for i in range(len(classes)):
    for j in range(len(classes)):
        dists[i,j] = mahalanobisDist(X[target==classes[i]],X[target==classes[j]])
matrix = ax2.imshow(dists,interpolation='none')
ax2.set_xticks(range(len(classes)))
ax2.set_xticklabels(classes,rotation=-90)
ax2.set_yticks(range(len(classes)))
ax2.set_yticklabels([i.split(' ')[0][0]+i.split(' ')[1][0] for i in classes])
fig.colorbar(matrix)

 #%% POINT CLOUD: DO SEXES SEPARATE OUT?========================================
from sklearn.decomposition import PCA
from sklearn.lda import LDA

#Truncate data feature:
data_sets = np.hsplit(data,fl[:-1]) #Split into feature sets
data_sets_trans,fittings = np.transpose(map(lambda x: autoID.modelBuilding.truncate_train_features(x,k=pcCutoff),data_sets)) #Truncate sets
data_trans = np.hstack(data_sets_trans) #Merge sets

classNames = [' '.join(i) for i in md[['Genus','Species','Sex']] ] #Combine values from groupClassBy into str for class name
specIDs = [autoID.modelBuilding._str2num(i) for i in md['Scan ID'].values] #Convert specimens IDs 'WRK-WS-#####' to ints from the ##### part of ID
id2label = dict(zip(specIDs,classNames))#Dict of specimen_IDs:class_labels
filename2label = {} #Dict of filenames:class_labels
for i in id2filename:
    try:
        filename2label[id2filename[i]] = id2label[i]
    except KeyError:
        filename2label[id2filename[i]] = None
for key, val in filename2label.items(): #Throw out specimens with None for ID
    if val is None: del filename2label[key]
del key,val,i,specIDs,classNames,id2label

#Do PCA:
pca=PCA(n_components=None,whiten=False)
pca.fit(data_trans)
var = pca.explained_variance_ratio_
pcaT=pca.transform(data_trans)
X = pcaT

fig,ax=plt.subplots(1,1,figsize=(18,6))
yy = np.array([filename2label[i] for i in target_fns])
classes=np.unique(yy)[:-1]#Leave out 'unknown' class
color=iter(plt.cm.nipy_spectral(np.linspace(0,1,len(classes))))
for cl in classes:
    if len(X[yy==cl])>8:
        c=next(color)
        m=list(np.mean(X[yy==cl],axis=0)[:2])#Mean location
        #Plot group rays:
        for x,y in X[yy==cl][:,:2]:
            ax.plot((x,m[0]),(y,m[1]),'-',c=c,alpha=0.3)
        #Plot points:
        ax.plot(X[yy==cl,0], X[yy==cl,1], 'o', alpha=0.3,
        c=c, label=cl)
        #Annotate with group label:
        ax.annotate(cl,#Annotate with class name
                xy=(0,0),xycoords='data', #dummy coords
                xytext=m,textcoords='data',
                color='k',fontsize=8#c#,weight='bold'##,
                )
ax.set_xlabel('PC 1 (%.2f %% of total variance)' % (var[0]*100))#,weight='bold',fontsize='xx-large');
ax.set_ylabel('PC 2 (%.2f %% of total variance)' % (var[1]*100))#,weight='bold',fontsize='xx-large');

#%%
#Do LDA:
lda = LDA(n_components=2)
ldaT = lda.fit(data_trans,target).transform(data_trans)
fig = plt.figure()
ax = fig.add_subplot(111,axisbg='k')
targets=np.unique(target)[:-1]
color=iter(plt.cm.rainbow(np.linspace(0,1,len(targets))))
for targ in targets:
    c=next(color)
    ax.plot(ldaT[target == targ,0],ldaT[target == targ,1],'o',c=c,label=targ)
    ax.annotate(targ,#Annotate with class name
            xy=(0,0),xycoords='data', #dummy coords
            xytext=np.mean(ldaT[target==targ],axis=0),textcoords='data',
            color=c,weight='heavy',fontsize='x-large',
            )
ax.set_xlabel('LD 1',fontsize='x-large');ax.set_ylabel('LD 2',fontsize='x-large');

#plt.legend(loc='upper right')
plt.show()
#%% VIEW CONFUSION MATRIX======================================================
#predicted = clf_model.predict(data_test_trans)
conf_arr = metrics.confusion_matrix(target_test, predicted)

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        try: p = float(j)/float(a)
        except ZeroDivisionError: p = 0
        tmp_arr.append(p)
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                interpolation='nearest')

width = len(conf_arr)
height = len(conf_arr[0])

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
targets=np.unique(target)
#Target abbreviations:
target_abbrev=[]
for targ in targets:
    f=targ.split(' ')#Split class name by ' '
    f=map(lambda x: x[:2],f)#Take first 2 letters
    f="".join(f)
    target_abbrev.append(f)

#Get counts from training set:
targTestCount = []
for targ in targets:
    targTestCount.append('%s (%d)' % (targ,len(target_train[target_train == targ])))

plt.xticks(range(width), target_abbrev,rotation='vertical')
plt.yticks(range(height), targTestCount)

#Source:http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
#%% NOTES: --------------------------------------------------------------------
'''
A Gaussian Naive Bayes classifier can be updated using .partial_fit() method!
This could be used to update a trained classifier model as new specimens are
added to an online dataset.  It's also good for breaking a really big dataset
into chunks & training/updating a Bayesian model with those chunks.  The only
limitation is that on first partial_fit, a list of all possible classes must be
provided, so specimens from new species couldn't be added with partial_fit.
Instead, the model would have to be retrained.

Use sklearn.preprocessing.label_binarize for one-for-each type classifiers?

'''
