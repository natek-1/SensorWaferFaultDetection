from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException
from src.utils import import_collection_as_dataframe
from src.constants import *
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer

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
X_train, X_test, y_train, y_test, _ = data_transformer.initiate_data_transformation(path)
model_trainer = ModelTrainer()
score, path = model_trainer.initiate_model_training(X_train=X_train, X_test=X_test,
                                                    y_train=y_train, 
                                                    y_test=y_test)
print(score)


