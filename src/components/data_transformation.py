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
