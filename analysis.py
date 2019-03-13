# Importing libraries we will use in the analysis
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split # splitting data
from sklearn.neighbors import KNeighborsRegressor    # regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV     # for grid search
from sklearn.pipeline import make_pipeline           # for making pipelines
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, make_scorer
import warnings
# Suppress warnings that we don't care about
warnings.filterwarnings(action='ignore', category=Warning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

pd.set_option('display.max_columns', 100)

from prep import * 

mean_absolute_error_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def pipeline(feat_train, labels_train, algo, param_grid, descriptor):
    train_features_small, validation_features, train_outcome_small, validation_outcome = train_test_split(
        feat_train,     # features
        labels_train,   # outcome
        test_size=0.20, # percentage of data to use as the test set
        random_state=11 # set a random state so it is consistent (not required!)
    )
    scaler = MinMaxScaler()
    kbest = SelectKBest(chi2)
    pipeline = make_pipeline(kbest, scaler, algo)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=mean_absolute_error_scorer)

    gsFit = grid_search.fit(train_features_small, train_outcome_small['norm_wage'].tolist())

    folds = KFold(n_splits=10, shuffle=True)

    predictions = cross_val_predict(gsFit, feat_train, labels_train, cv= folds)
    pd.DataFrame(labels_train).join(pd.DataFrame(predictions)).to_csv('model_predictions/' + descriptor + '.csv', index=False)

    return np.mean(cross_val_score(gsFit, feat_train, labels_train, cv= folds, scoring=mean_absolute_error_scorer))

print('Getting data ready...')
start_time = time.time()
df = prepared_df()
features = df.drop(['norm_wage', 'Nationality', 'Club'], axis=1)
labels = df.filter(['norm_wage'], axis=1)
numFeatures = len(features.columns)
print('Done (took %g seconds) Beginning predictions...'  % (time.time() - start_time))

regressors = [
    {
        'algo': GaussianNB(), 
        'param_grid': {
            'selectkbest__k': [2,4,10,20,numFeatures]
        },
        'descriptor': 'GaussianNB'
    },
    {
        'algo': KNeighborsRegressor(), 
        'param_grid': {
            'selectkbest__k': [2,4,10,20,numFeatures], 
            'kneighborsregressor__n_neighbors':range(1, 20), 
            'kneighborsregressor__weights':["uniform", "distance"]
        },
        'descriptor': 'KNeighborsRegressor'
    },
    {
        'algo': RandomForestRegressor(), 
        'param_grid': {
            'selectkbest__k': [2,4,10,20,numFeatures], 
            'randomforestregressor__n_estimators':range(10, 100)
        },
        'descriptor': 'RandomForestRegressor'
    },
    {
        'algo': DecisionTreeRegressor(), 
        'param_grid': {
            'selectkbest__k': [2,4,10,20,numFeatures], 
            'decisiontreeregressor__max_depth':range(100, 200)
        },
        'descriptor': 'DecisionTreeRegressor'
    },
    {
        'algo': MLPRegressor(), 
        'param_grid': {
            'selectkbest__k': [2,4,10,20,numFeatures], 
            'mlpregressor__learning_rate':['constant', 'invscaling', 'adaptive'],
            'mlpregressor__alpha': np.linspace(0.1,0.5,5)
        },
        'descriptor': 'MLPRegressor'
    }
]

for regressor in regressors:
    start_time = time.time()
    score = pipeline(
        features, 
        labels, 
        regressor['algo'], 
        regressor['param_grid'],
        regressor['descriptor']
    )
    print('%s took %g seconds and produced an average mean absolute error of %g'  % (regressor['descriptor'], time.time() - start_time, score))