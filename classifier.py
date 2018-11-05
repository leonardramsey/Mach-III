import os
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

score_conversion = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 0: 0}


# data clean
def data_cleanup(data):
    # number scaling and imputing
    if 'NutriScore' in data:
        data = data.drop('NutriScore', axis=1)
    data_num = data
    num_attribs = list(data_num)

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    data_num_tr_nd = num_pipeline.fit_transform(data_num)

    data_num_tr = pd.DataFrame(data_num_tr_nd, columns=data_num.columns,
                               index=list(data.index.values))

    print(data_num_tr.head())

    # prepared data
    data_prepared = data

    # update columns in processed data frame
    for feature in data_prepared:
        data_prepared[feature] = data_num_tr[feature]

    return data_prepared


def ml():
    # read in data
    print('Reading data...')
    with open('data.json') as f:
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

    # data info
    print('----------------------- Original Data -----------------------')
    print(label_info.head())
    print('----------------------- Original Data Info -----------------------')
    print(label_info.info())
    print(label_info.describe())
    print('----------------------- Correlations -----------------------')
    print(label_info.corr()['NutriScore'].sort_values(ascending=False))

    # data clean
    label_info = label_info.dropna(thresh=len(label_info) - 300, axis=1)
    label_info = label_info.drop(['Nutrition score  France', 'url', 'file_name', 'nutrition_label_src', 'sno'], axis=1)
    print('----------------------- Updated Data Info (after removing poor features) -----------------------')
    print(label_info.info())

    for feature in label_info:
        print(feature)
        try:
            label_info = label_info.apply(pd.to_numeric, errors='coerce')
        except Exception:
            print(str(feature) + " - unable to cast to numeric data type.")

    print('----------------------- Updated Data Info (after casting) -----------------------')
    print(label_info.info())
    print(
        '----------------------- Updated Correlations (after removing poor features and casting) -----------------------')
    print(label_info.corr()['NutriScore'].sort_values(ascending=False))

    train_set, test_set = train_test_split(label_info, test_size=0.2, random_state=48, stratify=label_info.NutriScore)

    # data split
    print('----------------------- X_train -----------------------')
    X_train = data_cleanup(train_set)
    print(X_train.head())
    print('----------------------- y_train -----------------------')
    y_train = train_set['NutriScore']
    print(y_train.head())
    print('----------------------- X_test -----------------------')
    X_test = data_cleanup(test_set)
    print(X_test.head())
    print('----------------------- y_test -----------------------')
    y_test = test_set['NutriScore']
    print(y_test.head())


    # training

    # testing

    # output


if __name__ == '__main__':
    ml()