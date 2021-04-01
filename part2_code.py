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

classifications = ['business', 'entertainment', 'politics', 'sport', 'tech']
training_data = []
test_data = []

for cl in classifications:
    path = 'D:/UNI/CMT316 - Applications of Machine Learning; Natural Language Processing and Computer Vision/Coursework 1/datasets_coursework1/bbc/' + cl
    for file in os.listdir(path):
        training_data.append([(open((path + '/' + str(file))).read()),cl])

shuffle(training_data)

for i in range(round(len(training_data) * 0.2)):
    test_data.append(training_data.pop(random.randint(0,len(training_data))))
