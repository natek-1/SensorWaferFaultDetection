from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from src.utils import import_collection_as_dataframe
from src.constants import *
import os
import sys
from src.pipeline.training_pipeline import TrainPipeline


#load_dotenv()

#logging.info("test1")
#try:
#    val = 1/0
#except Exception as e:
#    logging.error(CustomException(e, sys))


#print(import_collection_as_dataframe(MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME).head())



pipeline = TrainPipeline()
score = pipeline.run_pipeline()
print(score)