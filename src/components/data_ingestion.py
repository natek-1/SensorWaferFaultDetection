import os
import sys
from dataclasses import dataclass


import pandas as pd

from src.constants import MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME
from src.logger import logging
from src.exception import CustomException
from src.utils import import_collection_as_dataframe


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "data.csv")

class DataIngestion():
    def __init__(self):
        self.dataingestion_config: DataIngestion = DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            df: pd.DataFrame = import_collection_as_dataframe(MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME)

            logging.info("Exported collection as dataframe")

            os.makedirs(
                os.path.dirname(self.dataingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.dataingestion_config.raw_data_path, index=False, header=True)

            logging.info(
                f"Ingested data from mongodb to {self.dataingestion_config.raw_data_path}"
            )

            return self.dataingestion_config.raw_data_path
        
        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error)
            raise error

