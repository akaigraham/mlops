from datetime import datetime 
import time 
import random 
import logging 
import uuid 
import pytz 
import pandas as pd
import io 
import psycopg 
import joblib 
from prefect import task, flow 
from evidently.report import Report 
from evidently import ColumnMapping 
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics (
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin', 'rb') as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')

begin = datetime(2022,2,1,0,0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task 
def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost post=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)
            
