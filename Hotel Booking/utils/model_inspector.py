"""Model inspection and verification utility."""
import os
from utils.helpers import load_model
from logger.log_config import get_logger

logger = get_logger("model_inspector")


def inspect_model(model_path: str) -> None:
    """
    Load and inspect a saved scikit-learn model.
    
    Args:
        model_path: Path to the .joblib model file
        
    Raises:
        FileNotFoundError: If model file does not exist
        Exception: If model cannot be loaded
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        model = load_model(model_path)
        
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")
        
        # Inspect Logistic Regression specific attributes
        if hasattr(model, 'coef_'):
            logger.info(f"Coefficients shape: {model.coef_.shape}")
            logger.info(f"Number of features: {model.coef_.shape[1]}")
        
        if hasattr(model, 'intercept_'):
            logger.info(f"Intercept: {model.intercept_}")
        
        if hasattr(model, 'classes_'):
            logger.info(f"Classes: {model.classes_}")
        
        logger.info("Model inspection completed successfully!")
        return model
        
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


if __name__ == '__main__':
    # Standalone verification script
    model_path = os.path.join(os.getcwd(), 'artifacts', 'models', 'logistic_model.joblib')
    inspect_model(model_path)
