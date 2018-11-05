import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
# from sklearn.model_selection import train_test_split
# from pandas.plotting import scatter_matrix
# from sklearn.preprocessing import Imputer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_score, recall_score, accuracy_score
# from sklearn.metrics import roc_curve
# from sklearn.model_selection import cross_val_predict




def ml():
    # read in data
    label_info = pd.DataFrame(columns=["Calcium", "Carbohydrate", "Cholesterol", "Dietary fiber", "Energy",
        "Energy from fat", "Fat", "Iron", "NutriScore", "Nutrition score  France", "Potassium", "Proteins",
        "Salt", "Saturated fat", "Sodium", "Sugars", "Trans fat", "Vitamin A", "Vitamin C (ascorbic acid)",
        "Vitamin D", "file_name", "nutrition_label_src", "sno", "url"])
    with open('data.json') as f:
        data = json.load(f)
    i = 0
    for title in data:
        label_info.loc[i] = pd.Series(data[title])
    print('----------------------- Original Data + Info -----------------------')
    print(label_info.head())

    # data info
    print(label_info.info())
    print(label_info.describe())
    print('----------------------- Columns -----------------------')
    print(list(label_info))
    # data clean

    # data split

    # training

    # testing

    # output



if __name__ == '__main__':
    ml()