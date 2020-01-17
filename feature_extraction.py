from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#import mahotas
import cv2
import os
import h5py
import glob
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern

train_path = 'Train_preproc'
output_folder = "output"
#bins for colour histogram
bins = 8
random_seed = 9



def lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #parameters of lbp
    radius = 4
    n_points = 32
 
    lbp = local_binary_pattern(gray, n_points, radius, method = 'uniform')
    (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, n_points + 3),range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + (1e-7))
    return hist

# feature-extraction-1: Hu Moments-shape information
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-extraction-2: Haralick Texture-texture information
"""
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
"""

# feature-extraction-3: Color Histogram-color information
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], 
                                                   [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)

global_features = []
labels = []

for training_name in train_labels:
    path = glob.glob(train_path+"/"+training_name+"/*.tif")
    current_label = training_name
    
    x=0
    print ("[STATUS] processing folder: {}".format(current_label))
    for file in path:
        image = cv2.imread(file)

        #combination of different feature vectors
        global_feature = np.hstack([
        	fd_histogram(image),
        	#fv_haralick,
        	fd_hu_moments(image),
        	#lbp_features(image),
        	
        	])
        labels.append(current_label)
        global_features.append(global_feature)

    
    


print ("[STATUS] completed Global Feature Extraction...")
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

# normalization of feature vector
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

#print ("[STATUS] target labels: {}".format(target))
print( "[STATUS] target labels shape: {}".format(target.shape))

# saving files on h5 format. We can use it after
# this method most suitable while using jupyter
if(not os.path.exists(output_folder)):
    os.mkdir(output_folder)
h5f_data = h5py.File('output/data_512.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels_512.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()


models = []
models.append(('LR', LogisticRegression(random_state=random_seed, multi_class='auto', solver='lbfgs')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(random_state=random_seed)))
models.append(('RF', RandomForestClassifier(n_estimators=200, random_state=random_seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM_linear', SVC(kernel='linear')))
models.append(('SVM_rbf', SVC(kernel='rbf', C=10.0, gamma=2.0)))

results = []
names = []
scoring = "accuracy"


#reading of saved data
h5f_data = h5py.File('output/data_512.h5', 'r')
h5f_label = h5py.File('output/labels_512.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()



from sklearn.utils import shuffle
trainDataGlobal, trainLabelsGlobal = shuffle(np.array(global_features), np.array(global_labels), random_state=9)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))

trainDataGlobal, trainLabelsGlobal = shuffle(np.array(global_features), np.array(global_labels), random_state=9)

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


"""
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
"""


"""
from sklearn import metrics
for name, model in models:
    clf=model
    clf.fit(trainDataGlobal, trainLabelsGlobal)
    y_pred=clf.predict(testDataGlobal)
    msg = "%s: %f " % (name, metrics.accuracy_score(y_pred,testLabelsGlobal))
    print(msg)
"""