from pathlib import Path 
import pickle 
import pandas as pd
import numpy as np
import scipy 
from sklearn.feature_extraction import DictVectorizer 
from sklearn.metrics import mean_squared_error 
import mlflow
import xgboost as xgb
from prefect import flow, task

def read_data(filename: str) -> pd.DataFrame:
    """
    Read data from filename into pandas DataFrame.
    """
    df = pd.read_parquet(filename)
    
    # preprocess
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda x: x.total_seconds() / 60)
    
    # only include trips between 1 minute and 1 hour
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df