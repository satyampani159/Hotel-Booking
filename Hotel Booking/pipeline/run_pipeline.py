import os
import sys
import pandas as pd
from entity.config_entity import DataIngestionConfig, TrainingConfig
from components.data_ingestion import DataIngestion
from components.preprocessing import preprocess_pipeline
from components.trainer import Trainer
from components.visualizations import generate_all_visualizations
from constants import paths
from logger.log_config import get_logger

logger = get_logger("run_pipeline")


def run():
    logger.info("Starting pipeline run")

    data_cfg = DataIngestionConfig(data_dir=paths.DATA_DIR, data_file=paths.DATA_FILE)
    ingestion = DataIngestion(data_cfg)
    df_original = ingestion.load_data()

    df_processed = preprocess_pipeline(df_original)

    if 'is_canceled' not in df_processed.columns:
        logger.error('Target column `is_canceled` not found after preprocessing')
        return

    X = df_processed.drop('is_canceled', axis=1)
    y = df_processed['is_canceled']

    train_cfg = TrainingConfig()
    trainer = Trainer(train_cfg)
    results = trainer.train(X, y)

    logger.info(f"Training results: accuracy={results['accuracy']:.4f}, model_path={results['model_path']}")
    
    # Generate visualizations
    logger.info("Generating visualizations and reports...")
    try:
        final_rush = pd.DataFrame()  # Empty placeholder
        sorted_data = pd.DataFrame()  # Empty placeholder
        accuracy = results['accuracy']
        cm = results['confusion_matrix']
        cv_scores = results.get('cv_scores', None)
        
        generate_all_visualizations(
            df_original=df_original,
            df_processed=df_processed,
            final_rush=final_rush,
            sorted_data=sorted_data,
            cm=cm,
            accuracy=accuracy,
            cv_scores=cv_scores
        )
        logger.info("Visualizations generated successfully!")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise


if __name__ == '__main__':
    run()
