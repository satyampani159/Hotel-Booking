"""Comprehensive EDA and visualization module for hotel booking data."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger.log_config import get_logger

logger = get_logger("visualizations")

PLOTS_DIR = "artifacts/plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(filename: str, tight_layout=True):
    """Decorator to save plots to disk."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            ensure_plots_dir()
            result = func(*args, **kwargs)
            if tight_layout:
                plt.tight_layout()
            path = os.path.join(PLOTS_DIR, filename)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {path}")
            plt.close()
            return result
        return wrapper
    return decorator


def log_data_info(df: pd.DataFrame):
    """Log basic data info and save to file."""
    ensure_plots_dir()
    info_path = os.path.join(PLOTS_DIR, "01_data_shape_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Data Types:\n{df.dtypes}\n\n")
        f.write(f"Missing Values:\n{df.isnull().sum()}\n\n")
        f.write(f"First 5 Rows:\n{df.head()}\n")
    logger.info(f"Data info saved to {info_path}")
    return df


@save_plot("02_missing_values_heatmap.png")
def plot_missing_values(df: pd.DataFrame):
    """Plot missing values heatmap."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows (Sample)')


@save_plot("03_room_price_by_type.png")
def plot_room_price_boxplot(data: pd.DataFrame):
    """Boxplot: Price of room types per night by hotel."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='reserved_room_type', y='adr', hue='hotel', data=data)
    plt.title('Price of Room Types per Night and Person')
    plt.xlabel('Room Types')
    plt.ylabel('Price (EUR)')
    plt.xticks(rotation=45)


@save_plot("04_monthly_guest_trends.png")
def plot_monthly_guest_trends(final_rush: pd.DataFrame):
    """Line plot: Guest trends by month for resort vs city."""
    if final_rush.empty:
        logger.warning("final_rush dataframe is empty, skipping monthly trends plot")
        return
    plt.figure(figsize=(14, 7))
    plt.plot(final_rush.index, final_rush['no_of_guests_in_resort'], marker='o', label='Resort Hotel', linewidth=2)
    plt.plot(final_rush.index, final_rush['no_of_guests_city'], marker='s', label='City Hotel', linewidth=2)
    plt.title('Guest Trends by Month: Resort vs City Hotels')
    plt.xlabel('Month')
    plt.ylabel('Number of Guests')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)


@save_plot("05_adr_by_month_barplot.png")
def plot_adr_by_month(data: pd.DataFrame):
    """Barplot: Average room rate (ADR) by month with cancellation status."""
    plt.figure(figsize=(14, 6))
    sns.barplot(x='arrival_date_month', y='adr', hue='is_canceled', data=data)
    plt.title('Average Daily Rate (ADR) by Month and Cancellation Status')
    plt.xlabel('Arrival Month')
    plt.ylabel('ADR (EUR)')
    plt.xticks(rotation=45)


@save_plot("06_adr_by_month_boxplot.png")
def plot_adr_boxplot(data: pd.DataFrame):
    """Boxplot: ADR distribution by month."""
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='arrival_date_month', y='adr', hue='is_canceled', data=data)
    plt.title('ADR Distribution by Month (with Cancellation Status)')
    plt.xlabel('Arrival Month')
    plt.ylabel('ADR (EUR)')
    plt.ylim(0, 800)
    plt.xticks(rotation=45)


@save_plot("07_weekend_weekday_bookings.png")
def plot_weekend_weekday_breakdown(sorted_data: pd.DataFrame):
    """Stacked bar chart: Booking breakdown by weekend/weekday."""
    if sorted_data.empty:
        logger.warning("sorted_data is empty, skipping weekend/weekday plot")
        return
    plt.figure(figsize=(15, 8))
    sorted_data.plot(kind='bar', stacked=True, figsize=(15, 8))
    plt.title('Booking Breakdown: Weekend vs Weekday by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Bookings')
    plt.legend(title='Booking Type')
    plt.xticks(rotation=45)


@save_plot("08_lead_time_distribution.png")
def plot_lead_time_distribution(df: pd.DataFrame):
    """Distribution plot: Lead time."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['lead_time'], kde=True, bins=50)
    plt.title('Lead Time Distribution')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Frequency')


@save_plot("09_lead_time_by_cancellation.png")
def plot_lead_time_by_cancellation(df: pd.DataFrame):
    """KDE plot: Lead time distribution by cancellation status."""
    plt.figure(figsize=(12, 6))
    for cancel_status in [0, 1]:
        data_subset = df[df['is_canceled'] == cancel_status]['lead_time']
        label = 'Not Cancelled' if cancel_status == 0 else 'Cancelled'
        sns.kdeplot(data=data_subset, label=label, fill=True, alpha=0.5)
    plt.xlim(0, 500)
    plt.title('Lead Time Distribution by Cancellation Status')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Density')
    plt.legend()


@save_plot("10_adr_distribution_before.png")
def plot_adr_distribution_before(df: pd.DataFrame):
    """Distribution plot: ADR before outlier handling."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['adr'], kde=True, bins=50)
    plt.title('ADR Distribution (Before Outlier Handling)')
    plt.xlabel('ADR (EUR)')
    plt.ylabel('Frequency')


@save_plot("11_adr_distribution_after.png")
def plot_adr_distribution_after(df: pd.DataFrame):
    """Distribution plot: ADR after log transformation."""
    plt.figure(figsize=(12, 6))
    valid_adr = df['adr'].dropna()
    sns.histplot(valid_adr, kde=True, bins=50)
    plt.title('ADR Distribution (After Log Transformation)')
    plt.xlabel('Log(ADR)')
    plt.ylabel('Frequency')


@save_plot("12_correlation_heatmap.png")
def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap: Correlation matrix."""
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')


@save_plot("13_cancellation_correlation.png")
def plot_cancellation_correlation(df: pd.DataFrame):
    """Barplot: Feature correlation with cancellation."""
    plt.figure(figsize=(12, 8))
    if 'is_canceled' in df.columns:
        corr = df.corr()['is_canceled'].sort_values(ascending=False)
        corr.plot(kind='barh')
        plt.title('Feature Correlation with Cancellation')
        plt.xlabel('Correlation Coefficient')


@save_plot("14_confusion_matrix_heatmap.png")
def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Heatmap: Confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Cancelled', 'Cancelled'],
                yticklabels=['Not Cancelled', 'Cancelled'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def save_model_metrics(accuracy: float, cm: np.ndarray, cv_scores: np.ndarray = None):
    """Save model performance metrics to file."""
    ensure_plots_dir()
    metrics_path = os.path.join(PLOTS_DIR, "15_model_metrics_summary.txt")
    with open(metrics_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Negatives:  {cm[0,0]:6d}\n")
        f.write(f"  False Positives: {cm[0,1]:6d}\n")
        f.write(f"  False Negatives: {cm[1,0]:6d}\n")
        f.write(f"  True Positives:  {cm[1,1]:6d}\n\n")
        if cv_scores is not None and len(cv_scores) > 0:
            f.write(f"Cross-Validation Scores (10-fold):\n")
            f.write(f"  Mean: {cv_scores.mean():.4f}\n")
            f.write(f"  Std:  {cv_scores.std():.4f}\n")
            f.write(f"  Scores: {cv_scores}\n")
    logger.info(f"Metrics saved to {metrics_path}")


def generate_all_visualizations(df_original: pd.DataFrame, df_processed: pd.DataFrame,
                               final_rush: pd.DataFrame, sorted_data: pd.DataFrame,
                               cm: np.ndarray, accuracy: float, cv_scores: np.ndarray = None):
    """Generate all EDA and model evaluation plots."""
    logger.info("Generating all visualizations...")
    
    try:
        # Data info file
        try:
            log_data_info(df_original)
        except Exception as e:
            logger.warning(f"Could not generate data info: {e}")
        
        # Missing values heatmap (optional - can be expensive for large datasets)
        try:
            if len(df_original) < 50000:  # Only for smaller datasets
                plot_missing_values(df_original)
        except Exception as e:
            logger.warning(f"Could not generate missing values heatmap: {e}")
        
        # Room pricing plot
        try:
            data_completed = df_original[df_original['is_canceled'] == 0]
            if not data_completed.empty and 'reserved_room_type' in df_original.columns and 'adr' in df_original.columns:
                plot_room_price_boxplot(data_completed)
        except Exception as e:
            logger.warning(f"Could not generate room pricing plot: {e}")
        
        # Monthly guest trends
        try:
            if not final_rush.empty:
                plot_monthly_guest_trends(final_rush)
        except Exception as e:
            logger.warning(f"Could not generate monthly trends plot: {e}")
        
        # ADR by month plots
        try:
            if 'arrival_date_month' in df_original.columns and 'adr' in df_original.columns:
                plot_adr_by_month(df_original)
                plot_adr_boxplot(df_original)
        except Exception as e:
            logger.warning(f"Could not generate ADR plots: {e}")
        
        # Weekend/weekday breakdown
        try:
            if not sorted_data.empty:
                plot_weekend_weekday_breakdown(sorted_data)
        except Exception as e:
            logger.warning(f"Could not generate weekend/weekday plot: {e}")
        
        # Lead time plots
        try:
            if 'lead_time' in df_original.columns:
                plot_lead_time_distribution(df_original)
                plot_lead_time_by_cancellation(df_original)
        except Exception as e:
            logger.warning(f"Could not generate lead time plots: {e}")
        
        # ADR distribution before/after
        try:
            if 'adr' in df_processed.columns:
                plot_adr_distribution_before(df_original)
                plot_adr_distribution_after(df_processed)
        except Exception as e:
            logger.warning(f"Could not generate ADR distribution plots: {e}")
        
        # Correlation plots
        try:
            plot_correlation_heatmap(df_processed)
            plot_cancellation_correlation(df_processed)
        except Exception as e:
            logger.warning(f"Could not generate correlation plots: {e}")
        
        # Model evaluation plots
        try:
            plot_confusion_matrix(cm, "Logistic Regression - Confusion Matrix")
            save_model_metrics(accuracy, cm, cv_scores)
        except Exception as e:
            logger.warning(f"Could not generate model evaluation plots: {e}")
        
        logger.info("All visualizations generated successfully!")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise


