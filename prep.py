# Importing libraries we will use in the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler       # scaling data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split # splitting data
from sklearn.neighbors import KNeighborsRegressor    # regressor
from sklearn.model_selection import GridSearchCV     # for grid search
from sklearn.pipeline import make_pipeline           # for making pipelines

def prepared_df():
    # Data Preparation and Cleaning
    df = pd.read_csv('data.csv', index_col=0)

    df = df.fillna(method='ffill')
    df.head()

    dropped_features=['Club Logo', 'Flag', 'Photo', 'Name', 'Special', 'Body Type', 
                      'Real Face', 'Loaned From','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                      'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                      'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    df = df.drop(dropped_features, axis=1)
    return df

def fix_pos(row):
    pos = row['Position']
    forward = ['RS', 'LS', 'CF', 'RF', 'LF', 'ST', 'RW', 'LW']
    if pos in forward:
        return 3
    elif pos.endswith('M'):
        return 2
    elif pos.endswith('B'):
        return 1
    elif pos == 'GK':
        return 0
    

def enum_position():
    df = prepared_df()
    df['num_position'] = df.apply (lambda row: fix_pos(row), axis=1)
    df = df.drop('Position', axis=1)
    return df