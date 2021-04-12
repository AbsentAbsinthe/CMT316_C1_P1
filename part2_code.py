''' 
CMT316 - Applications of Machine Learning, Coursework 1 Part 2
Text classification

C1771290
Lewis Hemming
'''
#-----------------------------------------Imports---------------------------------------------------------------------------------------------------------------------
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import os
import random
from random import shuffle
#-----------------------------------------Data spliting---------------------------------------------------------------------------------------------------------------
classifications = ['business', 'entertainment', 'politics', 'sport', 'tech'] # These are the 5 possible classifications for the text
data = []
test_data = []

for cl in classifications:
    path = 'datasets_coursework1/bbc/' + cl
    for file in os.listdir(path):
        data.append([(open((path + '/' + str(file))).read()),cl])   # News Reports are read into a list of 2 objects, one containting the report as a string and one
                                                                    # containing the classification, this is then appended to a list containg all the data

shuffle(data) # The order of this data is then randomised to ensure each classification isn't all grouped together

for i in range(round(len(data) * 0.2)):
    test_data.append(data.pop(random.randint(0,(len(data)-1)))) # 20% of the data is then randomly chosen to be removed from the main data file and appended to the
                                                                # test set
Y_train = []

for article in data:
    Y_train.append(classifications.index(article[1])) # Y_train contains the correct classifications of all the training data


#-----------------------------------------Feature 1: Sparse Vector of all words--------------------------------------------------------------------------------------
print('Creating sparse vector...\n')

articles = []

for article in data: 
    articles.append(article[0])  # articles contains the string reports of all the training data

CVectorizer = CountVectorizer(stop_words='english') # A count vectorizer is created using english stopwords
X_count = CVectorizer.fit_transform(articles) 
X_train_f1 = X_count.toarray()

print('Feature 1 created\n')
#-----------------------------------------Feature 2: Weighted Vector of words------------------------------------------------------------------------------------------
print('Creating sparse vector...\n')

vectorizer = TfidfVectorizer(stop_words='english')
X_weighted = vectorizer.fit_transform(articles)

X_train_f2 = X_weighted.toarray()

print('Feature 2 created\n')
#------------------------------------------Feature 3: Using lists of common names and terms to create a vector--------------------------------------------------------
file_names=['business_terms.txt', 'celebrity_names.txt', 'political_parties.txt', 'sporting_terms.txt', 'technological_terms.txt']
file_words=[]

print('Processing text files...\n')

for i in range(len(file_names)):
    file = open(('feature_3/' + file_names[i]))
    words = ''
    tok_words = []
    while True:
        try:
            line = file.readline()
            words += line
        except UnicodeDecodeError:
            continue
        if not line:
            break
    file_words.append(words)

print('Creating Feature 3 Vector... \n')

f3_vectorizer = CountVectorizer(stop_words='english')
f3_vectorizer.fit(file_words)
X_count_f3 = f3_vectorizer.transform(articles)
X_train_f3 = X_count_f3.toarray()

print('Feature 3 created\n')
#-----------------------------------------Feature Selection-------------------------------------------------------------------------------------------------------

def feature_creation(X1, X2, X3):
    features = []
    for i in range(len(X1)):
        f_list = list(X1[i])
        f2_list = list(X2[i])
        f3_list = list(X3[i])
        for x in range(len(f2_list)):
            f_list.append(f2_list[x])
        for x in range(len(f3_list)):    
            f_list.append(f3_list[x])
        features.append(np.asarray(f_list))
    return features

X_train = feature_creation(X_train_f1, X_train_f2, X_train_f3)

print('Performing Feature selection...\n')

ch2_train = SelectKBest(chi2, k=1000)
X_train = ch2_train.fit_transform(X_train, Y_train)

print('Feature selection complete\n')
#-----------------------------------------SVM train and test/predict--------------------------------------------------------------------------------------------
print('Training SVM...\n')
clf = sklearn.svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, Y_train)
print('Training Complete\n')

print('Testing...\n')
true_classification = 0
false_classification = 0

Y_true = []
Y_predict = []
X_test_data = []

for article in test_data:
    Y_true.append(classifications.index(article[1]))
    X_test_data.append(article[0])

X_test_f1_trans = CVectorizer.transform(X_test_data)
X_test_f1 = X_test_f1_trans.toarray()

X_test_f2_trans = vectorizer.transform(X_test_data)
X_test_f2 = X_test_f2_trans.toarray()

X_test_f3 = f3_vectorizer.transform(articles)
X_test_f3 = X_count_f3.toarray()

X_test = feature_creation(X_test_f1, X_test_f2, X_test_f3)

X_test = ch2_train.transform(X_test)

predictions = clf.predict(X_test)
for i in range(len(predictions)):
    Y_predict.append(predictions[i])
    if predictions[i] == Y_true[i]:
        true_classification += 1
    else:
        false_classification += 1

Y_true_named = []
Y_predict_named = []

for a in range(len(Y_true)):
    Y_true_named.append(classifications[Y_true[a]])
    Y_predict_named.append(classifications[Y_predict[a]])
print('\n')
print('Testing Results:\n')
print('\n')
print(classification_report(Y_true_named, Y_predict_named))
print('\n  True results:    ' + str(true_classification) + '\n False results:    ' + str(false_classification))
