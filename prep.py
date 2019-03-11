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

    dropped_features=['Club Logo', 'Flag', 'Photo', 'Name', 'Special', 'Body Type', 
                      'Real Face', 'Loaned From','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                      'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                      'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
    df = df.drop(dropped_features, axis=1)
    df = df.dropna()
    df = prepare_heights(df)
    df = enum_position(df)
    df = split_work_rates(df)
    df = enum_workrate(df)
    df = enum_financials(df)
    return df

def prepare_heights(temp):
    heights = temp['Height'].str.split("\'")
    inches = [12 * int(i[0]) + int(i[1]) for i in heights]
    temp["Inches"] = inches
    return(temp)
    
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
    

def enum_position(df):
    df['num_position'] = df.apply (lambda row: fix_pos(row), axis=1)
    df = df.drop('Position', axis=1)
    return df


def split_work_rates(df):
    splits = df['Work Rate'].str.split("/ ")
    offensive = []
    defensive = []

    for i in splits:
        offensive.append(i[0])
        defensive.append(i[1])
    df['Offensive Work Rate'] = offensive
    df['Defensive Work Rate'] = defensive
    return df

def fix_workrate(row, pos):
    if row[pos + ' Work Rate'] == 'Low':
        return 0
    elif row[pos + ' Work Rate'] == 'Medium':
        return 1
    elif row[pos + ' Work Rate'] == 'High':
        return 2

def enum_workrate(df):
    df['Enum Defensive Work Rate'] = df.apply(lambda row: fix_workrate(row, 'Defensive'), axis=1)
    df['Enum Offensive Work Rate'] = df.apply(lambda row: fix_workrate(row, 'Offensive'), axis=1)

    return df

def fix_value(row, col):
    val = float(row[col][1:-1])
    if row[col].endswith('M'):
        return val*1000
    elif row[col].endswith('K'):
        return val

def enum_financials(df):
    df['norm_wage'] = df.apply (lambda row: fix_value(row, 'Wage'), axis=1)
    df['norm_value'] = df.apply (lambda row: fix_value(row, 'Value'), axis=1)
    df['norm_release'] = df.apply (lambda row: fix_value(row, 'Release Clause'), axis=1)
    df = df.drop(['Wage', 'Value', 'Release Clause'], axis=1)
    return df