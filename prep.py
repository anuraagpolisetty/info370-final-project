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
from datetime import datetime                        # for converting string into datetime type

def prepared_df():
    # Data Preparation and Cleaning
    df = pd.read_csv('data.csv', index_col=0)

    dropped_features=['Club Logo', 'Flag', 'Photo', 'Name', 'Special', 'Body Type', 
                      'Real Face', 'Loaned From','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                      'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                      'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause', 'Contract Valid Until']
    df = df.drop(dropped_features, axis=1)
    df = df.dropna()
    df = prepare_heights(df)
    df = enum_position(df)
    df = split_work_rates(df)
    df = enum_workrate(df)
    df = enum_financials(df)
    df = total_stats(df)
    df = power_foot(df)
    df = apply_difference(df)
    df = enum_weights(df)
    return df

def enum_strings(df):
    df = enum_nationality(df)
    df = enum_club(df)
    return df

def prepare_heights(temp):
    heights = temp['Height'].str.split("\'")
    inches = [12 * int(i[0]) + int(i[1]) for i in heights]
    temp['Inches'] = inches
    temp = temp.drop('Height', axis=1)
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
    df['enum_d_work_rate'] = df.apply(lambda row: fix_workrate(row, 'Defensive'), axis=1)
    df['enum_o_work_rate'] = df.apply(lambda row: fix_workrate(row, 'Offensive'), axis=1)

    df = df.drop(['Offensive Work Rate', 'Defensive Work Rate', 'Work Rate'], axis=1)
    return df

def fix_value(row, col):
    if row[col] == '€0':
        return 0
    val = float(row[col][1:-1])
    if row[col].endswith('M'):
        return val*1000
    elif row[col].endswith('K'):
        return val

def enum_financials(df):
    df['norm_wage'] = df.apply (lambda row: fix_value(row, 'Wage'), axis=1)
    df['norm_value'] = df.apply (lambda row: fix_value(row, 'Value'), axis=1)
    # df['norm_release'] = df.apply (lambda row: fix_value(row, 'Release Clause'), axis=1)
    df = df.drop(['Wage', 'Value'], axis=1)
    return df

def sum_stats(row):
    player_cols = ['Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']
    gk_cols = ['GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
    if(row['num_position'] == 0):
        return row[gk_cols].sum() * 5
    else:
        return row[player_cols].sum()

def total_stats(df):
    df['total_stats'] = df.apply(lambda row: sum_stats(row), axis=1)
    return df

def power_foot(df):
    df['power_foot_num'] = np.where(df['Preferred Foot'] == 'Right', 0, 1)
    df = df.drop('Preferred Foot', axis=1)
    return df

def string_indices(df, col):
    n = df[col].unique()
    nations_df = pd.DataFrame(n, columns=['value'])
    nations_df.sort_values('value', inplace=True)
    nations_df.reset_index(drop=True, inplace=True)
    nations_df.reset_index(level=0, inplace=True)
    return nations_df

def get_num_str(row, n_idx, col):
    return int(n_idx[n_idx.value == row[col]]['index'])

def enum_nationality(df):
    n_idx = string_indices(df, 'Nationality')
    df['num_nation'] = df.apply(lambda row: get_num_str(row, n_idx, 'Nationality'), axis=1)
    df = df.drop('Nationality', axis=1)
    return df

def enum_club(df):
    n_idx = string_indices(df, 'Club')
    df['num_club'] = df.apply(lambda row: get_num_str(row, n_idx, 'Club'), axis=1)
    df = df.drop('Club', axis=1)
    return df


def months_date_joined(row):
    datetime_object = datetime.strptime(row.Joined, '%b %d, %Y')
    today = datetime.strptime('Mar 11, 2019', '%b %d, %Y')
    difference = today - datetime_object
    return difference.days

def apply_difference(df):
    df['Days_at_Club'] = df.apply(lambda row: months_date_joined(row), axis=1)
    df = df.drop('Joined', axis=1)
    return df

def enum_weights(df):
    weights = df['Weight'].str[0:-3].astype(int)
    df['enum_weights'] = weights
    df = df.drop('Weight', axis=1)
    return df

