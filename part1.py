import numpy as np
import nltk
import sklearn
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
    if float(home_linesplit[-1]) < 30:
        Y_train.append(0)
    else:
        Y_train.append(1)

X_train_expensive=np.asarray(X_train)
Y_train_expensive=np.asarray(Y_train) # This step is really not necessary, but it is recommended to work with numpy arrays instead of Python lists.

svm_clf_expensive=sklearn.svm.SVC(kernel="linear",gamma='auto') # Initialize the SVM model
svm_clf_expensive.fit(X_train_expensive,Y_train_expensive) # Train the SVM model

test_path='D:/UNI/CMT316 - Applications of Machine Learning; Natural Language Processing and Computer Vision/Coursework 1/datasets_coursework1/real-state/test_full_Real-estate.csv'

test_dataset_file=open(test_path).readlines()

true_expensive = 0
false_expensive = 0
true_cheap = 0
false_cheap = 0

for home in test_dataset_file[1:]:
    test_home_linesplit=home.split(",")
    vector_test_home_features=np.zeros(len(test_home_linesplit)-2)
    for i in range(1, len(home_linesplit)-1):
        vector_test_home_features[i-1]=float(test_home_linesplit[i])
    prediction = svm_clf_expensive.predict([vector_test_home_features])
    if float(test_home_linesplit[-1]) < 30 and prediction == 0:
        true_cheap += 1
    elif float(test_home_linesplit[-1]) < 30 and prediction == 1:
        false_expensive += 1
    elif prediction == 1:
        true_expensive += 1
    else:
        false_cheap += 1

test_outcome=[true_expensive, false_expensive, true_cheap, false_cheap]

print('True Positive (True Expensive): ' + str(test_outcome[0]) + '\nFalse Positive (False Expensive): ' + str(test_outcome[1]) + '\nTrue Negative (True Cheap): ' + str(test_outcome[2]) + '\nFalse Negative (False Cheap): ' + str(test_outcome[3]))