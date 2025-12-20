import numpy as np
import pandas as pd
from logger.log_config import get_logger
from exception.custom_exception import CustomException

logger = get_logger("preprocessing")


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # drop agent and company as in notebook
        df = df.copy()
        if 'agent' in df.columns and 'company' in df.columns:
            df.drop(['agent', 'company'], axis=1, inplace=True)

        # fill country with mode
        if 'country' in df.columns:
            df['country'].fillna(df['country'].mode().iloc[0], inplace=True)

        # replace remaining nulls with 0 (as notebook did)
        df.fillna(0, inplace=True)

        # remove rows where adults, children and babies are all zero
        if set(['adults', 'children', 'babies']).issubset(df.columns):
            filt = (df['adults'] == 0) & (df['children'] == 0) & (df['babies'] == 0)
            df = df[~filt]

        return df
    except Exception as e:
        raise CustomException("Error in basic_cleaning", e)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        # family feature
        if set(['adults', 'children', 'babies']).issubset(df.columns):
            df['is_family'] = df.apply(lambda r: 1 if (r['adults'] > 0 and (r['children'] > 0 or r['babies'] > 0)) else 0, axis=1)
            df['total_customer'] = df['adults'] + df['children'] + df['babies']
            df['total_nights'] = df.get('stays_in_week_nights', 0) + df.get('stays_in_weekend_nights', 0)

        # deposit mapping
        if 'deposit_type' in df.columns:
            mapping = {'No Deposit': 0, 'Non Refund': 1, 'Refundable': 0}
            df['deposit_given'] = df['deposit_type'].map(mapping).fillna(0)

        # drop raw columns that won't be used
        drop_cols = [c for c in ['adults', 'children', 'babies', 'deposit_type'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        return df
    except Exception as e:
        raise CustomException("Error in feature_engineering", e)


def mean_encode_categoricals(df: pd.DataFrame, target: str = 'is_canceled') -> pd.DataFrame:
    try:
        df = df.copy()
        # select categorical columns
        cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        if target in df.columns:
            df_cat = df[cat_cols].copy()
            df_cat[target] = df[target]
            for col in cat_cols:
                enc = df_cat.groupby(col)[target].mean().to_dict()
                df[col] = df[col].map(enc)
        return df
    except Exception as e:
        raise CustomException("Error in mean_encode_categoricals", e)


def handle_outliers_log_transform(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    try:
        df = df.copy()
        if cols is None:
            cols = ['lead_time', 'adr']
        for col in cols:
            if col not in df.columns:
                continue
            # for adr, ensure non-negative before log
            series = df[col].copy()
            min_val = series.min()
            if pd.isnull(min_val):
                continue
            if min_val <= -1:
                # shift up
                shift = abs(min_val) + 1
                series = series + shift
            series = np.log1p(series.replace({np.nan: 0}))
            df[col] = series
        return df
    except Exception as e:
        raise CustomException("Error in handle_outliers_log_transform", e)


def select_and_drop_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        features_to_drop = ['reservation_status', 'reservation_status_date', 'arrival_date_year',
                            'arrival_date_week_number', 'stays_in_weekend_nights', 'arrival_date_day_of_month']
        to_drop = [c for c in features_to_drop if c in df.columns]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
        return df
    except Exception as e:
        raise CustomException("Error in select_and_drop_features", e)


def preprocess_pipeline(df: pd.DataFrame, generate_plots: bool = True) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline with logging.
    
    Args:
        df: Input dataframe
        generate_plots: Whether to generate EDA plots during preprocessing
    
    Returns:
        Preprocessed dataframe
    """
    from components.output_reports import print_dataframe_info, print_data_cleaning_summary
    
    # Log initial state
    print_dataframe_info(df.copy(), stage="00 - Initial Data")
    
    df_before = df.copy()
    df = basic_cleaning(df)
    print_data_cleaning_summary(df_before, df, "After Basic Cleaning")
    
    df_before = df.copy()
    df = feature_engineering(df)
    print_data_cleaning_summary(df_before, df, "After Feature Engineering")
    
    df_before = df.copy()
    df = mean_encode_categoricals(df)
    print_data_cleaning_summary(df_before, df, "After Mean Encoding")
    
    df_before = df.copy()
    df = handle_outliers_log_transform(df)
    print_data_cleaning_summary(df_before, df, "After Outlier Handling")
    
    df_before = df.copy()
    df = select_and_drop_features(df)
    # drop remaining na
    df.dropna(inplace=True)
    print_data_cleaning_summary(df_before, df, "After Feature Selection & Null Removal")
    
    # Log final state
    print_dataframe_info(df.copy(), stage="Final Preprocessed Data")
    
    return df

