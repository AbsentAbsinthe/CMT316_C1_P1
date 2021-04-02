import numpy as np
import nltk
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import operator
import os
import random
from random import shuffle
import requests

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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


lemmatizer = nltk.stem.WordNetLemmatizer()

def get_list_tokens(string):
    sentence_split=nltk.tokenize.sent_tokenize(string)
    list_tokens=[]
    for sentence in sentence_split:
        list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
        list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens

# Add all possible stopwords to ensure that they aaren't counted in the most frequent words
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

dict_word_frequency={}
for article in data:
    sentence_tokens=get_list_tokens(article[0])
    for word in sentence_tokens:
        if word in stopwords: continue
        if word not in dict_word_frequency: dict_word_frequency[word]=1
        else: dict_word_frequency[word]+=1

sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:1000]
i=0

frequent_words=[]
for word,frequency in sorted_list:
    frequent_words.append(word)

def get_vector_text(list_vocab,string):
    vector_text=np.zeros(len(list_vocab))
    list_tokens_string=get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i]=list_tokens_string.count(word)
    return vector_text

X_train = []
Y_train = []

for article in data:
    article_vector=get_vector_text(frequent_words,article[0])
    X_train.append(article_vector)
    if article[1] == 'business':
        Y_train.append(0)
    elif article[1] == 'entertainment':
        Y_train.append(1)
    elif article[1] == 'politics':
        Y_train.append(2)
    elif article[1] == 'sport':
        Y_train.append(3)
    else:
        Y_train.append(4)

X_train_np=np.asarray(X_train)
Y_train_np=np.asarray(Y_train)

clf = sklearn.svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_np, Y_train_np)

true_classification = 0
false_classification = 0

for article in test_data:
    article_vector=get_vector_text(frequent_words,article[0])
    prediction = clf.predict([article_vector])
    if classifications[prediction[0]] == article[1]:
        true_classification += 1
    else:
        false_classification += 1

print('True results : ' + str(true_classification) + '\nFalse results: ' + str(false_classification))


