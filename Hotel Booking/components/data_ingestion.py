import os
import pandas as pd
from typing import Optional
from entity.config_entity import DataIngestionConfig
from logger.log_config import get_logger
from exception.custom_exception import CustomException

logger = get_logger("data_ingestion")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        try:
            path = self.config.data_path
            logger.info(f"Loading data from: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found at {path}")
            df = pd.read_csv(path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
            
            # Print loaded data information
            from components.output_reports import print_dataframe_info
            print_dataframe_info(df.copy(), stage="Raw Data Loaded")
            
            return df
        except Exception as e:
            raise CustomException("Failed during data ingestion", e)
