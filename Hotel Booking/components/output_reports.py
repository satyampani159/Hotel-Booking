"""Module for generating text-based reports and console outputs."""
import os
import io
import pandas as pd
from logger.log_config import get_logger

logger = get_logger("output_reports")

PLOTS_DIR = "artifacts/plots"


def ensure_reports_dir():
    """Ensure reports directory exists."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def print_and_save_text(title: str, content: str, filename: str = None):
    """Print content to console and optionally save to file."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(content)
    
    if filename:
        ensure_reports_dir()
        filepath = os.path.join(PLOTS_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(f"{title}\n")
            f.write("=" * 80 + "\n")
            f.write(content)
        logger.info(f"Saved report to {filepath}")


def print_dataframe_info(df: pd.DataFrame, stage: str = "Data"):
    """Print comprehensive DataFrame information."""
    ensure_reports_dir()
    
    output = []
    output.append(f"\n{'=' * 80}")
    output.append(f"STAGE: {stage}")
    output.append(f"{'=' * 80}\n")
    
    # Shape
    output.append(f"Shape: {df.shape} (rows, columns)\n")
    
    # First 5 rows
    output.append("First 5 Rows:")
    output.append(df.head().to_string())
    output.append("")
    
    # Data types
    output.append("Data Types:")
    output.append(df.dtypes.to_string())
    output.append("")
    
    # Missing values
    output.append("Missing Values:")
    output.append(df.isnull().sum().to_string())
    output.append("")
    
    # Descriptive statistics
    output.append("Descriptive Statistics:")
    output.append(df.describe().to_string())
    output.append("")
    
    # Column names
    output.append("All Columns:")
    output.append(str(list(df.columns)))
    output.append("")
    
    content = "\n".join(output)
    print(content)
    
    # Save to file
    filename = f"00_dataframe_{stage.lower().replace(' ', '_')}.txt"
    filepath = os.path.join(PLOTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    logger.info(f"Saved DataFrame info to {filepath}")


def print_data_cleaning_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, stage: str):
    """Print summary of data cleaning operations."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append(f"DATA CLEANING: {stage}")
    output.append(f"{'=' * 80}\n")
    
    rows_before = len(df_before)
    rows_after = len(df_after)
    rows_dropped = rows_before - rows_after
    
    cols_before = len(df_before.columns)
    cols_after = len(df_after.columns)
    cols_dropped = cols_before - cols_after
    
    output.append(f"Rows: {rows_before} -> {rows_after} (dropped: {rows_dropped})")
    output.append(f"Columns: {cols_before} -> {cols_after} (dropped: {cols_dropped})")
    output.append(f"Nulls before: {df_before.isnull().sum().sum()}")
    output.append(f"Nulls after: {df_after.isnull().sum().sum()}\n")
    
    content = "\n".join(output)
    print(content)


def print_categorical_features(df: pd.DataFrame, categorical_cols: list):
    """Print value counts for categorical features."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append("CATEGORICAL FEATURES - VALUE COUNTS")
    output.append(f"{'=' * 80}\n")
    
    for col in categorical_cols:
        if col in df.columns:
            output.append(f"\n{col}:")
            output.append(df[col].value_counts().to_string())
            output.append("")
    
    content = "\n".join(output)
    print(content)


def print_numerical_features(df: pd.DataFrame, numerical_cols: list):
    """Print statistics for numerical features."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append("NUMERICAL FEATURES - STATISTICS")
    output.append(f"{'=' * 80}\n")
    
    for col in numerical_cols:
        if col in df.columns:
            output.append(f"\n{col}:")
            output.append(df[col].describe().to_string())
            output.append("")
    
    content = "\n".join(output)
    print(content)


def print_feature_importance(features: list, title: str = "Selected Features"):
    """Print important features list."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append(title)
    output.append(f"{'=' * 80}\n")
    output.append(f"Number of features: {len(features)}\n")
    output.append("Features:")
    for i, feat in enumerate(features, 1):
        output.append(f"  {i}. {feat}")
    output.append("")
    
    content = "\n".join(output)
    print(content)


def print_model_training_summary(X_train_shape, X_test_shape, model_name: str, accuracy: float, cm):
    """Print model training summary."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append(f"MODEL TRAINING: {model_name}")
    output.append(f"{'=' * 80}\n")
    
    output.append(f"Training set shape: {X_train_shape}")
    output.append(f"Test set shape: {X_test_shape}")
    output.append(f"Model: {model_name}")
    output.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    output.append("Confusion Matrix:")
    output.append(str(cm))
    output.append("")
    
    content = "\n".join(output)
    print(content)
    
    # Save to file
    filepath = os.path.join(PLOTS_DIR, "99_model_metrics_summary.txt")
    with open(filepath, 'w') as f:
        f.write(content)
    logger.info(f"Model metrics saved to {filepath}")


def print_cross_validation_summary(cv_scores: list, mean_score: float, std_score: float):
    """Print cross-validation summary."""
    output = []
    output.append(f"\n{'=' * 80}")
    output.append("CROSS-VALIDATION RESULTS")
    output.append(f"{'=' * 80}\n")
    
    output.append(f"Number of folds: {len(cv_scores)}")
    output.append(f"Scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
    output.append(f"Mean CV Accuracy: {mean_score:.4f} ({mean_score*100:.2f}%)")
    output.append(f"Std Dev: {std_score:.4f}\n")
    
    content = "\n".join(output)
    print(content)
