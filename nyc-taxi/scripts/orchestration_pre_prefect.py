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

def add_features(
    df_train: pd.DataFrame, 
    df_val: pd.DataFrame   
) -> tuple(
    [scipy.sparse._csr.csr_matrix,
     scipy.sparse._csr.csr_matrix,
     np.ndarray,
     np.ndarray,
     DictVectorizer]
):
    """
    Add features to the model
    """
    # combine pickup and drop off into one feature
    df_train['PU_DO'] = df_train['PULocationID'] + '-' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '-' + df_val['DOLocationID']
    
    # specify cat and numerical features
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dv = DictVectorizer()
    
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    
    return X_train, X_val, y_train, y_val, dv