from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from src.utils import import_collection_as_dataframe
from src.constants import *
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer

#load_dotenv()

#logging.info("test1")
#try:
#    val = 1/0
#except Exception as e:
#    logging.error(CustomException(e, sys))


#print(import_collection_as_dataframe(MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME).head())

data_ingestion = DataIngestion()
path = data_ingestion.initial_data_ingestion()
data_transformer = DataTransformer()
val = data_transformer.initiate_data_transformation(path)
print(len(val))
print(val)

