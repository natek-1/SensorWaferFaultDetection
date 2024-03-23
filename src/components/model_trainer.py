import os
import sys
from dataclasses import dataclass


import pandas as pd
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import r2_score


from src.utils import evaluate_models, load_obj, save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")

class CustomModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, X):
        preprocessed_data = self.preprocessor.predict(X)

        return self.trained_model_object.predict(preprocessed_data)
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, X_train, X_test, y_train, y_test, preprocessor_path):
        try:
            logging.info("starting the model training step")

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC()
            }

            report = evaluate_models(X_train_full=X_train, y_train_full=y_train, models=models)

            best_model_score = max(sorted(report.values()))

            if best_model_score < 0.6:
                raise Exception("No best model found")
            
            logging.info(f"There was a model found with a score of {best_model_score}")
            
            best_model_name = list(report.keys())[
                list(report.values()).index(best_model_score)
            ]
            logging.info(f"model name is {best_model_name}")
            
            best_model = models[best_model_name]

            # Training the best model on dataset

            best_model.fit(X_train, y_train)


            y_pred = best_model.predict(X_test)

            score = r2_score(y_test, y_pred)

            logging.info(f"testing the model on the full train dataset gives a score of {score}")

            preprocessing_obj = load_obj(preprocessor_path)

            custom_model = CustomModel(
                preprocessor=preprocessing_obj,
                model=best_model,
            )

            save_object(obj=custom_model, file_path=self.model_trainer_config.model_path)

            return score, self.model_trainer_config.model_path

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error
