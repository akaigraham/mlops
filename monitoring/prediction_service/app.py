import os 
import pickle

import requests 
from flask import Flask 
from flask import request 
from flask import jsonify 

from pymongo import MongoClient 

MODEL_FILE = os.getenv('MODEL_FILE', 'lin_reg.bin')
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
