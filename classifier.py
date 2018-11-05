import os
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
# from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.metrics import roc_curve
# from sklearn.model_selection import cross_val_predict

score_conversion = {'A' : 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}

# def data_cleanup(data):
#     # number scaling and imputing
#     cat_attribs = ['ISO country code', 'Country', 'Sub-national region','World region']
#     mpi_num = data.drop(cat_attribs, axis=1)
#     num_attribs = list(mpi_num)
#     mpi_cat = data.drop(num_attribs, axis=1)
#
#     num_pipeline = Pipeline([
#          ('imputer', Imputer(strategy="median")),
#          ('std_scaler', StandardScaler()),
#     ])
#
#     mpi_num_tr_nd = num_pipeline.fit_transform(mpi_num)
#
#     mpi_num_tr = pd.DataFrame(mpi_num_tr_nd, columns=mpi_num.columns,
#                               index = list(data.index.values))
#
#     print(mpi_num_tr.head())
#     # encoding of categories/text
#     lab_encoder = LabelEncoder()
#     cat_encoder = OneHotEncoder(sparse=False)
#
#     # label encoder
#     mpi_enc_cc = lab_encoder.fit_transform(data['ISO country code'])
#     mpi_enc_country = lab_encoder.fit_transform(data['Country'])
#     mpi_enc_snr = lab_encoder.fit_transform(data['Sub-national region'])
#     mpi_enc_wr = lab_encoder.fit_transform(data['World region'])
#
#     # cat encoder (OneHotEncoder)
#     mpi_cat_cc_hot = cat_encoder.fit_transform(mpi_enc_cc.reshape(-1,1))
#     mpi_cat_country_hot = cat_encoder.fit_transform(mpi_enc_country.reshape(-1,1))
#     mpi_cat_snr_hot = cat_encoder.fit_transform(mpi_enc_snr.reshape(-1,1))
#     mpi_cat_wr_hot = cat_encoder.fit_transform(mpi_enc_wr.reshape(-1,1))
#
#     # prepared data
#     mpi_data_prepared = data
#     # categories
#     mpi_data_prepared['ISO country code'] = mpi_cat_cc_hot
#     mpi_data_prepared['Country'] = mpi_cat_country_hot
#     mpi_data_prepared['Sub-national region'] = mpi_cat_snr_hot
#     mpi_data_prepared['World region'] = mpi_cat_wr_hot
#     # numbers
#     mpi_data_prepared['MPI Regional'] = mpi_num_tr['MPI Regional']
#     mpi_data_prepared['Headcount Ratio Regional'] = mpi_num_tr['Headcount Ratio Regional']
#     mpi_data_prepared['Intensity of deprivation Regional'] = mpi_num_tr['Intensity of deprivation Regional']
#     return mpi_data_prepared

def ml():
    # read in data
    columns = []
    print('Reading data...')
    with open('data.json') as f:
        data = json.load(f)
    # get the columns
    for title in data:
        for feature in data[title]:
            if feature not in columns:
                columns.append(feature)
    label_info = pd.DataFrame(columns=columns)
    i = 0
    print(data)
    # build dataframe
    for title in data:
        if 'NutriScore' in data[title]:
            data[title]['NutriScore'] = score_conversion[data[title].get('NutriScore', 0)]
            label_info.loc[i] = pd.Series(data[title])
        i+=1
    print('----------------------- Original Data + Info -----------------------')
    print(label_info)

    # data info
    print(label_info.info())
    print(label_info.describe())
    print('----------------------- Columns -----------------------')
    print(list(label_info))
    # label
    print('----------------------- Correlations -----------------------')
    print(label_info.corr())
    # data clean
    label_info = label_info.drop(['file_name', 'url', 'nutrition_label_src'], axis=1)

    # label_info_prepared = data_cleanup(label_info)

    # data split

    # training

    # testing

    # output



if __name__ == '__main__':
    ml()