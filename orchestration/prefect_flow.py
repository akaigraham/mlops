import pandas as pd
import pickle 

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error 

import xgboost as xgb 

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials 
from hyperopt.pyll import scope 

import mlflow 

from prefect import flow, task

@task
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


