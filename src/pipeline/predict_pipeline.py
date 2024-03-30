import os
import sys
from dataclasses import dataclass

import pandas as pd
from flask import request


from src.logger import logging
from src.exception import CustomException
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN
from src.utils import load_obj


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
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files["file"]
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file)

            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def predict(self, features):
        try:

            model = load_obj(self.prediction_pipeline_config.model_file_path)

            preprocessor = load_obj(obj_path=self.prediction_pipeline_config.preprocessor_file_path)

            transformed_x = preprocessor.transform(features)
            preds = model.predict(transformed_x)

            return preds
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def get_predicted_dataframe(self, input_data_frame_path: str):
        try:
            
            input_dataframe: pd.DataFrame = pd.read_csv(input_data_frame_path)

            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)

            input_dataframe[TARGET_COLUMN] = [pred for pred in predictions]
            
            target_column_mapping = {0: "bad",
                                     1: "good"}
            
            input_dataframe[TARGET_COLUMN] = input_dataframe[TARGET_COLUMN].map(target_column_mapping)

            os.makedirs(self.prediction_pipeline_config.predicted_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_file()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error


