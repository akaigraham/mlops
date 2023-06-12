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

