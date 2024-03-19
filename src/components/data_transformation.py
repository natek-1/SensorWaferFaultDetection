import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprecessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformer():

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):
        try:
            replace_nan = lambda X: np.where(X == "na", np.nan, X)

            replace_nan_step = ("nan_replace", FunctionTransformer(replace_nan))
            imputer_step = ("imputer", KNNImputer(n_neighbors=3))
            scaler_step = ("scaler_step", RobustScaler())

            preprocessor = Pipeline(
                steps = [
                    replace_nan_step,
                    imputer_step,
                    scaler_step
                ]
            )

            logging.info("Preprocessor created")

            return preprocessor
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
    
    @staticmethod
    def column_with_no_sd(df: pd.DataFrame):
        """
        Returns a list of columns names who are having zero standard deviation.
        """
        column_names = []
        #find numeric columns
        num_column = [col for col in df.columns if df[col].dtype != 'O']
        for col in num_column:
            if df[col].std() == 0:
                column_names.append(col)
        return column_names
    
    @staticmethod
    def get_redundant_cols(df: pd.DataFrame, missing_thresh=0.7):
        """
        Returns a list of columns having missing values more than certain thresh.
        """
        # empty value threshold
        ratio_per_col = df.isna().sum().div(df.shape[0])
        redundant_cols = list(ratio_per_col[ratio_per_col > missing_thresh].index)
        return redundant_cols
    
    def initiate_data_transformation(self, data_path):
        try:
            pass
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
