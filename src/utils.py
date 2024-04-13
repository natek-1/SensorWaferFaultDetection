import os
import sys

import pickle
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from src.logger import logging
from src.exception import CustomException


def import_collection_as_dataframe(db_name, collection_name):
    '''
    finds the information that is in the mongo database presented in the enviroment variable
    and returns the information in the form of a dataframe
    '''

    try:
        load_dotenv()
        url = os.getenv("MONGO_DB_LINK")

        client = MongoClient(url)
        client.admin.command('ping')
        logging.info("Pinged your deployment. You successfully connected to MongoDB!")

        collection = client[db_name][collection_name] 
        df = pd.DataFrame(list(collection.find()))
        logging.info("the dataframe was created")
        df.drop("_id", axis=1, inplace=True)
        return df
    
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error


def save_object(file_path, obj):
    '''
    save the object that is give into the file path that is specified
    makes sure that the directory is available before saving 
    '''

    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error


def evaluate_models(X_train_full, y_train_full, models: dict):
    '''
    function that take in the different models and dataset already split.
    And test each model to see which one perform

    returns a dictionary with the key as the model name and the value as thier score
    '''

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    report = {}

    for model_name, model_obj in models.items():
        model_obj.fit(X_train, y_train)

        y_val_pred = model_obj.predict(X_val)
        score = r2_score(y_true=y_val, y_pred=y_val_pred)

        report[model_name] = score

    return report


def load_obj(obj_path):
    '''
    returns the obj that is saved in the path present in the obj_path parameter
    '''
    try:
        obj = None
        with open(obj_path, "rb") as file:
            obj = pickle.load(file)
        return obj
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error
    

