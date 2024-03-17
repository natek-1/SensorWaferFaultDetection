import os
import sys

import pandas as pd
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

        client = MongoClient(url, server_api=ServerApi('1'))
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
