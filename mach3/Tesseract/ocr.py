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

score_conversion = {'A': 2, 'B': 2, 'C': 1, 'D': 0, 'E': 0, 0: -1}

# data clean
def data_cleanup(data):
    # number scaling and imputing
    data_num = data.copy()
    if 'NutriScore' in data_num:
        data_num = data_num.drop('NutriScore', axis=1)

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    data_num_tr_nd = num_pipeline.fit_transform(data_num)

    data_num_tr = pd.DataFrame(data_num_tr_nd, columns=data_num.columns,
                               index=list(data_num.index.values))

    # # prepared data
    # data_prepared = data
    # print('----------------------- data_prepared --------------------')
    # print(len(data_prepared.columns))
    # print(data_prepared.columns)
    # print('----------------------- data_num --------------------')
    # print(len(data_num_tr.columns))
    # print(data_num_tr.columns)
    #
    # # update columns in processed data frame
    # for feature in data_prepared:
    #     data_prepared[feature] = data_num_tr[feature]

    return data_num_tr

def classifier_metrics(y_test, y_pred):
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test,
                                                                                                             y_pred,
                                                                                                             average='weighted')

def ml():
    # read in data
    print('Reading data...')
    with open('../data/data_v3.json') as f:
        data = json.load(f)

    # get the columns
    columns = []
    for title in data:
        for feature in data[title]:
            if feature not in columns:
                columns.append(feature)
    label_info = pd.DataFrame(columns=columns)

    i = 0
    # build dataframe
    for title in data:
        if 'NutriScore' in data[title]:
            data[title]['NutriScore'] = score_conversion[data[title].get('NutriScore', 0)]
            label_info.loc[i] = pd.Series(data[title])
            i += 1
    label_info['NutriScore'] = pd.to_numeric(label_info['NutriScore'])

    # data clean
    label_info = label_info.fillna(0)
    label_info = label_info.drop(['Nutrition score  France', 'url', 'file_name', 'nutrition_label_src', 'sno'], axis=1)
    full_data = label_info.copy()
    print(len(full_data.columns))
    #     label_info = label_info.dropna(thresh=len(label_info) - 750, axis=1)
    label_info = label_info.dropna(thresh=25, axis=0)
    print(len(label_info.columns))
    difference = set(full_data.columns).difference(set(label_info.columns))
    print('------------------------------- Difference ---------------------------')
    print(difference)

    for feature in label_info:
        try:
            label_info = label_info.apply(pd.to_numeric, errors='coerce')
        except Exception:
            print(str(feature) + " - unable to cast to numeric data type.")

    train_set, test_set = train_test_split(label_info, test_size=0.2, random_state=48, stratify=label_info.NutriScore)

    # data split
    # print('----------------------- X_train -----------------------')
    X_train = data_cleanup(train_set)
    # print(X_train.head())
    # print('----------------------- y_train -----------------------')
    y_train = train_set['NutriScore']
    # print(y_train.head())
    # print('----------------------- X_test -----------------------')
    X_test = data_cleanup(test_set)
    # print(X_test.head())
    # print('----------------------- y_test -----------------------')
    y_test = test_set['NutriScore']
    # print(y_test.head())

    rnd_clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
    rnd_clf2.fit(X_train, y_train)
    y_pred_rnd_clf2 = rnd_clf2.predict(X_test)

    rnd_clf3 = RandomForestClassifier(n_estimators=500, random_state=42)
    rnd_clf3.fit(X_train, y_train)
    y_pred_rnd_clf3 = rnd_clf3.predict(X_test)

    # SVM
    c_list = [.1, 1, 10]
    gamma_list = [.01, .1, 1, 5]
    hyperparams = [(g, c) for g in gamma_list for c in c_list]
    best_gamma_C = None
    best_precision_rbf = 0

    # RBF SVC
    count2 = 1
    for gamma, C in hyperparams:
        rbf_svm_clf = SVC(kernel="rbf", gamma=gamma, C=C, probability=True)
        rbf_svm_clf.fit(X_train, y_train)
        y_pred_rbf_svm_clf = rbf_svm_clf.predict(X_test)
        accuracy, precision, recall = classifier_metrics(y_test, y_pred_rbf_svm_clf)
        if precision > best_precision_rbf:
            best_precision_rbf = precision
            best_gamma_C = (gamma, C)
        count2 += 1

    # best RBF SVC
    rbf_svm_clf = SVC(kernel="rbf", gamma=best_gamma_C[0], C=best_gamma_C[1], probability=True)
    rbf_svm_clf.fit(X_train, y_train)

    # voting
    voting_clf = VotingClassifier(estimators=[('rf2', rnd_clf2), ('rf3', rnd_clf3), ('rbf', rbf_svm_clf)],
                                  voting='hard')
    voting_clf.fit(X_train, y_train)
    return voting_clf, X_train.columns

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
   #  features = ['Energy from fat','Sodium','Monounsaturated fat','Trans fat','Potassium','Fat','Dietary fiber','Proteins','Calcium','Vitamin A','Iron','Cholesterol','Salt','Energy','Sugars','Carbohydrate','Saturated fat','Vitamin C (ascorbic acid)','Alcohol','Polyunsaturated fat','Vitamin B1 (Thiamin)','Vitamin B6 (Pyridoxin)','Vitamin B9 (Folic acid)','Vitamin B2 (Riboflavin)','Phosphorus','Vitamin B12 (cobalamin)','Magnesium','Vitamin D','Vitamin B3 / Vitamin PP (Niacin)','Zinc','\"Fruits, vegetables and nuts (minimum)\"','Pantothenic acid / Pantothenate (Vitamin B5)','&nbsp;  Lactose','Vitamin E','Cocoa (minimum)','Omega 3 fatty acids','Folates (total folates)','Copper','Chromium','\"Fruits, vegetables and nuts (estimate from ingredients list)\"','Erythritol','&nbsp;  Oleic acid (18:1 n-9)','Selenium','Omega 6 fatty acids','Caffeine','&nbsp;  Soluble fiber','&nbsp;  Insoluble fiber','FIBRA DIÉTETICA','Biotin','Vitamin K','Omega3 DHA (Docosahexaenoic Acid)','Omega3 Other','Omega3 EPA (Eicosapentaenoic Acid)','added sugars','Guarana Seed Extract','Taurine','FIBRA DIETÉTICA','Manganese','Molybdenum','Sugar alcohols (Polyols)','Iodine','Carbon footprint / CO2 emissions','Ecological footprint']
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
    ret_string += "\n"
    for feature in output:
        if feature != last_feat:
            ret_string += str(output[feature])+","
    ret_string += str(output[last_feat])+"\n"
    print('-------------------------- ret_string --------------------------')
    print(ret_string)
    f = open("Tesseract/output.csv","w")
    f.write(ret_string)
    f.close()

def prediction(img):
    get_test_data(img)
    data = pd.read_csv('Tesseract/output.csv')
    data = data.apply(pd.to_numeric, errors='coerce')
    print('-------------------------- Data columns: --------------------------')
    print(len(data.columns))
    print(data.columns)
    # clf = pickle.load(open("Tesseract/hoosfit.pkl","rb"))
    clf, model_columns = ml()
    print('-------------------------- X_train columns: --------------------------')
    print(len(model_columns))
    print(model_columns)
    difference = set(data.columns).difference(set(model_columns))
    print('------------------------------- Difference ---------------------------')
    print(difference)
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

    print('-------------------------- Data_prepared columns: --------------------------')
    print(len(data_prepared.columns))
    print(data_prepared.columns)
    score = clf.predict(data_prepared)
    print(score[0])
    return score[0]
