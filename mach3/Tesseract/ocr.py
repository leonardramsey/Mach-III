# coding=utf-8
import re
from collections import Counter
# from autocorrect import spell
import boto3
import os
import traceback
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('Tesseract/big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def get_test_data(img):
    client=boto3.client('rekognition')
    with open(img, 'rb') as image:
        response = client.detect_text(Image={'Bytes': image.read()})
    #response=client.detect_text(Image={'S3Object':{'Bucket':bucket,'Name':img}})
    features = ['Calcium', 'Carbohydrate', 'Cholesterol', 'Dietary fiber', 'Energy',
   'Fat', 'Iron', 'Potassium', 'Proteins', 'Salt', 'Saturated fat',
   'Sodium', 'Trans fat', 'Vitamin A', 'Vitamin B12 (cobalamin)',
   'Vitamin B2 (Riboflavin)', 'Vitamin C (ascorbic acid)', 'Vitamin D',
   'Energy from fat', 'Monounsaturated fat', 'Polyunsaturated fat',
   'Sugars', 'Vitamin B1 (Thiamin)', 'Vitamin B3 / Vitamin PP (Niacin)',
   'Vitamin B6 (Pyridoxin)', 'Vitamin B9 (Folic acid)', 'Zinc',
   'Phosphorus', 'Alcohol', 'Folates (total folates)', 'Magnesium',
   '&nbsp;  Insoluble fiber', 'Biotin', 'Chromium', 'Copper', 'Iodine',
   'Manganese', 'Molybdenum',
   'Pantothenic acid / Pantothenate (Vitamin B5)', 'Selenium', 'Vitamin E',
   'Vitamin K', '&nbsp;  Soluble fiber', 'Cocoa (minimum)',
   'Sugar alcohols (Polyols)',
   '\"Fruits, vegetables and nuts (estimate from ingredients list)\"',
   '\"Fruits, vegetables and nuts (minimum)\"', 'FIBRA DIETÉTICA', 'Starch',
   'Caffeine', 'Erythritol', 'Allulose', 'Omega 3 fatty acids',
   'Omega 6 fatty acids', '&nbsp;  Lactose',
   'Carbon footprint / CO2 emissions', 'Ecological footprint',
   'added sugars', '&nbsp;  Alpha-linolenic acid / ALA (18:3 n-3)']
    textDetections=response['TextDetections']
    output = {}
    for nutrition in features:
        output[nutrition] = 0
    matches = [0]*len(features)
    for text in textDetections:
        if(text["Type"] == 'LINE'):
            fullline = correction(text['DetectedText'])
            line = re.sub(r'[,|-]',r'',fullline)
            matches = [0] * len(features)
            ll1 = str(line).lower().split(" ")
            for i in range(len(features)):
               ll2 = features[i].lower().split(" ")
               matches[i] = len(list(set(ll1).intersection(ll2)))
            if max(matches)>0:
                pattern = r"\d+.*\d* *m?g"
                numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
                rx = re.compile(numeric_const_pattern, re.VERBOSE)
                try:
                    tot_val = re.search(pattern, line).group()
                    val = rx.findall(tot_val)
                    output[features[matches.index(max(matches))]] = float(val[0])
                except:
                    continue
    ret_string = ""
    last_feat = features[-1]
    for feature in output:
        if feature != last_feat:
            ret_string += feature+","
    ret_string += last_feat
    ret_string += features[-1]+"\n"
    for feature in output:
        if feature != last_feat:
            ret_string += str(output[feature])+","
    ret_string += str(output[last_feat])+"\n"

    f = open("Tesseract/output.csv","w")
    f.write(ret_string)
    f.close()

def prediction(img):
    get_test_data(img)
    data = pd.read_csv('Tesseract/output.csv')
    print(data)
    clf = pickle.load(open("Tesseract/hoosfit.pkl","rb"))
    if 'NutriScore' in data:
        data = data.drop('NutriScore', axis=1)
    data_num = data
    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    data_num_tr_nd = num_pipeline.fit_transform(data_num)
    data_num_tr = pd.DataFrame(data_num_tr_nd, columns=data_num.columns,
                           index=list(data.index.values))
    data_prepared = data
    for feature in data_prepared:
        data_prepared[feature] = data_num_tr[feature]

    score = clf.predict(data_prepared)
    print(score)
    return score
