import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Load the processed CKD dataset.

    Args:
        filepath (str): Path to processed data

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def explore_data(df):
    """
    Explore the dataset and print basic statistics.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    try:
        # Drop the empty unknown_26 column if it exists
        if 'unknown_26' in df.columns:
            df = df.drop(columns=['unknown_26'])
            logger.info("Dropped empty 'unknown_26' column")

        # Basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Column names: {df.columns.tolist()}")

        # Check data types
        logger.info("Data types:")
        logger.info(df.dtypes)

        # Check missing values
        missing_values = df.isna().sum()
        logger.info("Missing values per column:")
        logger.info(missing_values)

        # Distribution of target variable
        if 'class' in df.columns:
            logger.info("Target variable distribution:")
            logger.info(df['class'].value_counts())

        # Fix column names for wbcc and rbcc to match the dataset documentation
        if 'wbcc' in df.columns:
            df = df.rename(columns={'wbcc': 'wc'})
            logger.info("Renamed 'wbcc' to 'wc' (white blood cell count)")

        if 'rbcc' in df.columns:
            df = df.rename(columns={'rbcc': 'rc'})
            logger.info("Renamed 'rbcc' to 'rc' (red blood cell count)")

        # Create output directory for plots
        os.makedirs('../../data/plots', exist_ok=True)

        return df

    except Exception as e:
        logger.error(f"Error exploring data: {str(e)}")
        raise


def clean_data(df):
    """
    Clean the dataset by handling missing values and inconsistent entries.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        # Create a copy to avoid modifying the original
        df_cleaned = df.copy()

        # Convert categorical columns to lowercase for consistency
        categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
                            'class']
        for col in categorical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].str.lower() if df_cleaned[col].dtype == 'object' else df_cleaned[col]

        # Convert numerical columns to appropriate types
        numerical_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in numerical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        # Handle missing values
        # For numerical columns, fill missing values with median
        for col in numerical_cols:
            if col in df_cleaned.columns:
                median_value = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_value)
                logger.info(f"Filled missing values in '{col}' with median: {median_value}")

        # For categorical columns, fill missing values with mode
        for col in categorical_cols:
            if col in df_cleaned.columns:
                mode_value = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                logger.info(f"Filled missing values in '{col}' with mode: {mode_value}")

        # Clean up the 'class' column
        if 'class' in df_cleaned.columns:
            # Ensure 'class' is binary: 'ckd' or 'notckd'
            df_cleaned['class'] = df_cleaned['class'].apply(
                lambda x: 'ckd' if x.lower() in ['ckd', 'yes', '1', 'true'] else 'notckd' if isinstance(x, str) else x
            )
            logger.info("Standardized 'class' column values")

        # Check for any remaining missing values
        remaining_missing = df_cleaned.isna().sum().sum()
        logger.info(f"Remaining missing values after cleaning: {remaining_missing}")

        return df_cleaned

    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise


def create_disease_stage_features(df):
    """
    Create features related to disease stages based on clinical knowledge of CKD.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    try:
        # Create a copy to avoid modifying the original
        df_features = df.copy()

        # Create GFR estimation (simplified MDRD formula)
        # This is a medical formula to estimate kidney function
        if 'sc' in df_features.columns and 'age' in df_features.columns:
            # Make sure values are numeric
            df_features['sc'] = pd.to_numeric(df_features['sc'], errors='coerce')
            df_features['age'] = pd.to_numeric(df_features['age'], errors='coerce')

            # Simplified GFR calculation (actual formula is more complex)
            # GFR = 175 × (Scr)^-1.154 × (Age)^-0.203 × 0.742 [if female] × 1.212 [if Black]
            # We'll use a simplified version since we don't have all variables
            mask = (df_features['sc'].notna()) & (df_features['age'].notna()) & (df_features['sc'] > 0)
            df_features.loc[mask, 'gfr'] = 175 * (df_features.loc[mask, 'sc'] ** -1.154) * (
                        df_features.loc[mask, 'age'] ** -0.203)

            # Fill missing GFR values with median
            median_gfr = df_features['gfr'].median()
            df_features['gfr'] = df_features['gfr'].fillna(median_gfr)

            logger.info(
                f"Created GFR column with range: {df_features['gfr'].min():.2f} to {df_features['gfr'].max():.2f}")

            # Create CKD stages based on GFR
            conditions = [
                df_features['gfr'] >= 90,
                (df_features['gfr'] >= 60) & (df_features['gfr'] < 90),
                (df_features['gfr'] >= 45) & (df_features['gfr'] < 60),
                (df_features['gfr'] >= 30) & (df_features['gfr'] < 45),
                (df_features['gfr'] >= 15) & (df_features['gfr'] < 30),
                df_features['gfr'] < 15
            ]
            choices = [1, 2, 3, 3.5, 4, 5]

            df_features['ckd_stage'] = np.select(conditions, choices, default=np.nan)
            logger.info("Created CKD stage column based on GFR")
            logger.info(f"CKD stage distribution: {df_features['ckd_stage'].value_counts().sort_index()}")

        # Create risk score based on multiple factors
        risk_factors = ['htn', 'dm', 'cad', 'ane', 'pe']

        # Initialize risk score
        df_features['risk_score'] = 0

        # Add 1 to risk score for each positive risk factor
        for factor in risk_factors:
            if factor in df_features.columns:
                # Check if the factor value indicates positive
                if df_features[factor].dtype == 'object':
                    mask = df_features[factor].str.lower().isin(['yes', 'y', 'present', 'true', '1']).fillna(False)
                else:
                    mask = df_features[factor].fillna(0) > 0
                df_features.loc[mask, 'risk_score'] += 1

        logger.info(
            f"Created risk score with range: {df_features['risk_score'].min()} to {df_features['risk_score'].max()}")
        logger.info(f"Risk score distribution: {df_features['risk_score'].value_counts().sort_index()}")

        # Create progression risk category
        df_features['progression_risk'] = 'low'
        df_features.loc[
            (df_features['risk_score'] >= 2) & (df_features['risk_score'] < 4), 'progression_risk'] = 'medium'
        df_features.loc[df_features['risk_score'] >= 4, 'progression_risk'] = 'high'

        logger.info(f"Created progression risk categories: {df_features['progression_risk'].value_counts()}")

        return df_features

    except Exception as e:
        logger.error(f"Error creating disease stage features: {str(e)}")
        raise


def save_processed_data(df, output_path):
    """
    Save processed dataframe to CSV.

    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the processed data
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved processed data to {output_path}")

    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise


if __name__ == "__main__":
    # Define paths
    data_path = "../../data/processed/ckd_processed.csv"
    output_path = "../../data/processed/ckd_features.csv"

    # Load data
    df = load_data(data_path)

    # Explore data
    df = explore_data(df)

    # Clean data
    df_cleaned = clean_data(df)

    # Create disease stage features
    df_features = create_disease_stage_features(df_cleaned)

    # Save processed data
    save_processed_data(df_features, output_path)

    logger.info("Data exploration and feature engineering completed")