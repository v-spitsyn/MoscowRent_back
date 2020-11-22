import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer



PROJECT_FOLDER_PATH = pathlib.Path().resolve().parents[1]
FEATURES_PATH = PROJECT_FOLDER_PATH / 'data/processed/features_2020-08-01_2020-08-14.csv'
MODELS_FOLDER_PATH = PROJECT_FOLDER_PATH / 'models'


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def form_train_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=['pub_date'], index_col=0)
    train_df = df[(df['pub_date'] < pd.to_datetime('2020-08-13')) &
                  (((df['n_rooms'] < 3) & (df['y'] < 2)) |
                   ((df['n_rooms'] == 3) & (df['y'] < 2.5)) |
                   ((df['n_rooms'] > 3) & (df['y'] < 3.7))
                   )]
    X = train_df.drop(columns=['pub_date', 'y'])
    y = train_df['y']
    return X, y

def save_model(model, name):
    model_path = MODELS_FOLDER_PATH / f'{name}.pickle'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=7, criterion='mae', 
                           n_jobs=3, random_state=1)

X, y = form_train_data()
rf.fit(X, y)
save_model(rf, 'rf_2020-08-11')
