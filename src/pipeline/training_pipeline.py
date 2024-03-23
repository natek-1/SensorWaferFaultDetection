import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:

    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformer()
        self.model_trainer = ModelTrainer()
    
    def run_pipeline(self):
        try:
            data_ingestion = DataIngestion()
            path = data_ingestion.initial_data_ingestion()
            data_transformer = DataTransformer()
            X_train, X_test, y_train, y_test, preprocessor_path = data_transformer.initiate_data_transformation(path)
            model_trainer = ModelTrainer()
            score, path = model_trainer.initiate_model_training(X_train=X_train, X_test=X_test,
                                                                y_train=y_train, 
                                                                y_test=y_test, preprocessor_path=preprocessor_path)
            return score
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error