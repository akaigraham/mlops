import datetime 
import time 
import random 
import logging
import uuid 
import pytz 
import pandas as pd 
import io 
import psycopg 
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

# SQL for creating table 
create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    value1 integer, 
    value2 varchar,
    value3 float
)
"""

reference_data = pd.read_parquet('./data/reference.parquet')
with open('./models/lin_reg.bin', 'rb') as f_in:
    model = joblib.load(f_in)
    
raw_data = pd.read_parquet('./data/green_tripdata_2022-02.parquet')
begin = datetime.datetime(2022, 2, 1, 0, 0)

def prep_db():
    
    # access database with the following information
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)
    
    return None
            
def calculate_dummy_metrics_postgresql(curr):
    value1 = rand.randint(0,1000)
    value2 = str(uuid.uuid4())
    value3 = rand.random()
    
    curr.execute(
        "insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
        (datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
    )
    return None

def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 100):
            with conn.cursor() as curr:
                calculate_dummy_metrics_postgresql(curr)
            
            new_send = datetime.datetime.now() 
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")
            
if __name__ == '__main__':
    main()