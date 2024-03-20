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

def replace_nan(X):
    return np.where(X == "na", np.nan, X)

class DataTransformer():

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_object(self):
        try:

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
            logging.info("initiating data transformation")
            raw_df = pd.read_csv(data_path)

            # drop useless columns (feature selection)
            redundant_cols = DataTransformer.get_redundant_cols(raw_df)
            no_std_cols = DataTransformer.column_with_no_sd(raw_df)

            cols_to_drop =  no_std_cols  + redundant_cols
            cols_to_drop.append('Unnamed: 0')
            #print(cols_to_drop)

            X, y = raw_df.drop(columns=cols_to_drop, axis=1).iloc[:,:-1], raw_df.iloc[:,-1]

            logging.info("Applied feature selection by removeing redundant columns and columns with no std")

            logging.info(f"remaining columns are: {raw_df.drop(columns=cols_to_drop, axis=1).columns}")
            logging.info("split data into X and y")

            # Transofrom the using the preprocessor 
            preprocessor = self.get_transformation_object()

            X_trans = preprocessor.fit_transform(X)
            target_column_mapping = {1: 0, -1: 1}
            y_trans = y.map(target_column_mapping)
            logging.info("transormed the elements in the dataset")

            # resample the values of for X and y

            resampler = SMOTETomek(sampling_strategy="auto", random_state=42)
            X_res, y_res = resampler.fit_resample(X_trans, y_trans)

            logging.info("resampled the columns to match similar length")
            logging.info(f"Number of data point classified as 1, {len(y_res[y_res == 1])}")
            logging.info(f"Number of data point classified as 0, {len(y_res[y_res == 0])}")            

            # Create trans test split
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=42, test_size=0.2)
            logging.info("Data split into tran test split")

            # Save preprocessed object as designated path in config
            save_object(file_path=self.data_transformation_config.preprecessor_obj_path,
                        obj=preprocessor)

            # return X_train, X_test, y_train, y_test
            return  X_train, X_test, y_train, y_test, self.data_transformation_config.preprecessor_obj_path
            

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
