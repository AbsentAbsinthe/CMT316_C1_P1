import numpy as np
import nltk
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import operator
import os
import random
from random import shuffle
import requests

nltk.download('punkt') #Used to download necessary libraries
nltk.download('wordnet')
nltk.download('stopwords')

#-----------------------------------------Data Handling and Test/Train split---------------------------------------------------------------------------
classifications = ['business', 'entertainment', 'politics', 'sport', 'tech'] 
data = []
test_data = []

for cl in classifications:
    path = 'D:/UNI/CMT316 - Applications of Machine Learning; Natural Language Processing and Computer Vision/Coursework 1/datasets_coursework1/bbc/' + cl
    for file in os.listdir(path):
        data.append([(open((path + '/' + str(file))).read()),cl])

shuffle(data)


for i in range(round(len(data) * 0.2)):
    test_data.append(data.pop(random.randint(0,(len(data)-1))))

#-----------------------------------------Data Preprocessing -------------------------------------------------------------------------------------
lemmatizer = nltk.stem.WordNetLemmatizer()

def get_list_tokens(string):
    sentence_split=nltk.tokenize.sent_tokenize(string)
    list_tokens=[]
    for sentence in sentence_split:
        list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
        list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


# Add all possible stopwords to ensure that they aren't counted in the most frequent words
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add('\n')
stopwords.add('.')
stopwords.add(',')
stopwords.add('--')
stopwords.add('``')
stopwords.add('"')
stopwords.add('(')
stopwords.add(')')
stopwords.add('-')
stopwords.add(')')
stopwords.add("'")
stopwords.add("''")
stopwords.add(':')

X_train = []
Y_train = []

for article in data:
    Y_train.append(classifications.index(article[1]))

def get_vector_text(list_vocab,string):
    vector_text=np.zeros(len(list_vocab))
    list_tokens_string=get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i]=list_tokens_string.count(word)
    return vector_text


#-----------------------------------------Feature 1: Sparse Vector of 750 most common words---------------------------------------------------------
dict_word_frequency={}
for article in data:
    sentence_tokens=get_list_tokens(article[0])
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency: dict_word_frequency[word]=1
        else: dict_word_frequency[word]+=1

sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)

frequent_words=[]
for word,frequency in sorted_list:
    frequent_words.append(word)

#-----------------------------------------Feature 2: Weighted Vector of 750 words with highest significance----------------------------------------
articles = []

for article in data:
    articles.append(article[0])

vectorizer = TfidfVectorizer(stop_words='english')
X_train_2 = vectorizer.fit_transform(articles)

feature_names = vectorizer.get_feature_names()

ch2_feature = SelectKBest(chi2, k=750)
X_train_2 = ch2_feature.fit_transform(X_train_2, Y_train)

feature_names = [feature_names[i] for i in ch2_feature.get_support(indices=True)]

same = 0
diff = 0

for name in feature_names:
    if name in frequent_words:
        same += 1
    else:
        diff += 1

print('\nUsing weighted word frequency vs counted word frequency gave ' + str(same)+ ' of the same words and ' + str(diff) + ' different words\n')

all_features = feature_names
for word in frequent_words:
    all_features.append(word) 

print(len(all_features))


for article in data:
    article_vector=get_vector_text(all_features,article[0])
    X_train.append(article_vector)

#------------------------------------------Feature 3: Names---------------------------------------------------------------------------------------
file_names=['business_terms.txt', 'celebrity_names.txt', 'political_parties.txt', 'sporting_terms.txt', 'technological_terms.txt']
file_words=[]

for i in range(len(file_names)):
    file = open(file_names[i])
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
    tok_words.append(get_list_tokens(word))
    file_words.append(words)

for article in data:
    vector_f3 = np.zeros(len(file_names))
    for i in range(len(file_words)):
        total_word_appearances = 0
        file_vector = get_vector_text(file_words[i],article[0])
        for w in file_vector:
            total_word_appearances += w
        vector_f3[i] = total_word_appearances
    print(X_train[1])
    X_train[data.index(article)] = np.asarray(list(X_train[data.index(article)]).append(vector_f3))

#-----------------------------------------Feature Selection---------------------------------------------------------------------------------------

ch2_train = SelectKBest(chi2, k=600)
X_train = ch2_train.fit_transform(X_train, Y_train)


#-----------------------------------------SVM train and test/predict------------------------------------------------------------------------------
X_train_np=np.asarray(X_train)
Y_train_np=np.asarray(Y_train)

clf = sklearn.svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, Y_train)

true_classification = 0
false_classification = 0

X_test = []
Y_true = []
Y_predict = []

for article in test_data:
    article_vector=get_vector_text(all_features,article[0])
    X_test.append(article_vector)
    Y_true.append(classifications.index(article[1]))

X_test = ch2_train.transform(X_test)

predictions = clf.predict(X_test)
for i in range(len(predictions)):
    Y_predict.append(predictions[i])
    if predictions[i] == Y_true[i]:
        true_classification += 1
    else:
        false_classification += 1
print('\n')
print(classification_report(Y_true, Y_predict))
print('\n  True results:    ' + str(true_classification) + '\n False results:    ' + str(false_classification))

