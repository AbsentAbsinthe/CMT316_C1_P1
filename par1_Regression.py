import numpy as np
import nltk
import sklearn
from sklearn.metrics import mean_squared_error as mse
import operator
import requests

train_path='D:/UNI/CMT316 - Applications of Machine Learning; Natural Language Processing and Computer Vision/Coursework 1/datasets_coursework1/real-state/train_full_Real-estate.csv'

dataset_file=open(train_path).readlines()

X_train=[]
Y_train=[]
for home in dataset_file[1:]:
    home_linesplit=home.split(",")
    vector_home_features=np.zeros(len(home_linesplit)-2)
    for i in range(1, len(home_linesplit)-1):
        vector_home_features[i - 1]=float(home_linesplit[i])
    X_train.append(vector_home_features)
    Y_train.append(float(home_linesplit[-1]))

X_train_expensive=np.asarray(X_train)
Y_train_expensive=np.asarray(Y_train)

svm_clf_expensive=sklearn.svm.SVR(kernel="poly",degree=3,cache_size=2000)
svm_clf_expensive.fit(X_train_expensive,Y_train_expensive) 

test_path='D:/UNI/CMT316 - Applications of Machine Learning; Natural Language Processing and Computer Vision/Coursework 1/datasets_coursework1/real-state/test_full_Real-estate.csv'

test_dataset_file=open(test_path).readlines()

predictions = []
actual = []

for home in test_dataset_file[1:]:
    test_home_linesplit=home.split(",")
    vector_test_home_features=np.zeros(len(test_home_linesplit)-2)
    for i in range(1, len(home_linesplit)-1):
        vector_test_home_features[i-1]=float(test_home_linesplit[i])
    prediction = svm_clf_expensive.predict([vector_test_home_features])
    predictions.append(prediction)
    actual.append(float(test_home_linesplit[-1]))

predictions_array=np.asarray(predictions)
actual_array=np.asarray(actual)

mserror = mse(actual_array, predictions_array)

print("Mean squared error: " + str(mserror))