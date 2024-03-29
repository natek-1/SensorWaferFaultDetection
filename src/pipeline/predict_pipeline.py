import os
import sys
from dataclasses import dataclass

import pandas as pd
from flask import request


from src.logger import logging
from src.exception import CustomException
from src.constants import ARTIFACTS_DIR


@dataclass
class PredictPipelineConfig:
    predicted_output_dirname: str = "predictions"
    prediction_filename: str = "predicted_file.csv"

    model_file_path :str = os.path.join(ARTIFACTS_DIR, "model.pkl")
    preprocessor_file_path: str = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

    prediction_file_path = os.path.join(predicted_output_dirname. prediction_filename)

class PredictionPipeline:

    def __init__(self, request: request):
        self.request = request
        self.prediction_pipeline_config = PredictPipelineConfig()


    def save_input_file(self):
        try:
            pass
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def predict(self, features):
        try:
            pass
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def get_predicted_dataframe(self, input_data_frame_path: pd.DataFrame):
        try:
            pass
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def run_pipeline(self):
        try:
            pass
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error


