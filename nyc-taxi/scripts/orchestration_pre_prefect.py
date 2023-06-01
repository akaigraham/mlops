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

def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer
) -> None:
    """
    Train a model with the best hyperparams and write
    everything out.
    """
    
    # start mlflow run
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        
        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42, 
        }
        
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=20
        )
        
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metrics('rmse', rmse)
        
        Path("../models").mkdir(exist_ok=True)
        with open("../models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow") 
    
    return None

def main_flow(
    train_path: str = "../data/green_tripdata_2022-01.parquet",
    val_path: str = "../data/green_tripdata_2022-02.parquet"
) -> None:
    """
    The main training pipeline
    """
    
    # mlflow settings
    mlflow.set_tracking_uri("http://ec2-44-201-134-182.compute-1.amazonaws.com:5000")
    mlflow.set_experiment("nyc-taxi")
    
    # load data
    df_train = read_data(train_path)
    df_val = read_data(val_path)
    
    # transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # train
    train_best_model(X_train, X_val, y_train, y_val, dv)
    
    return None