import pandas as pd
import os
import logging
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ckd_data(filepath):
    """
    Load the Chronic Kidney Disease dataset in ARFF format from UCI.
    This function manually parses the ARFF file to handle inconsistencies.

    Args:
        filepath (str): Path to the dataset file

    Returns:
        pd.DataFrame: Loaded and basic preprocessed dataframe
    """
    try:
        # Read the file as text
        with open(filepath, 'r') as file:
            content = file.readlines()

        # Parse the file to extract attribute names and data
        attributes = []
        data_started = False
        data_lines = []

        for line in content:
            line = line.strip()
            if not line or line.startswith('%'):
                continue

            # Check if we've reached the data section
            if line.lower() == '@data':
                data_started = True
                continue

            # Parse attributes
            if line.lower().startswith('@attribute') and not data_started:
                parts = line.split(None, 2)  # Split into 3 parts: @attribute, name, type
                if len(parts) >= 3:
                    attr_name = parts[1].strip("'")
                    attributes.append(attr_name)

            # Collect data lines
            elif data_started:
                data_lines.append(line)

        logger.info(f"Found {len(attributes)} attributes in the ARFF header")

        # Process the data lines to determine the maximum number of columns
        data_rows = []
        max_columns = 0

        for i, line in enumerate(data_lines):
            # Split by comma, respecting quoted values
            values = next(csv.reader([line], delimiter=',', quotechar='"'))
            # Clean up spaces and handle '?' for missing values
            cleaned_values = [v.strip() if v.strip() != '?' else pd.NA for v in values]
            data_rows.append(cleaned_values)

            # Track maximum number of columns
            max_columns = max(max_columns, len(cleaned_values))

            # Log any rows with unexpected column counts
            if len(cleaned_values) != len(attributes):
                logger.warning(f"Row {i + 1} has {len(cleaned_values)} columns (expected {len(attributes)})")
                logger.warning(f"Row content: {cleaned_values}")

        logger.info(f"Maximum columns in any row: {max_columns}")

        # Extend attribute list if needed
        if max_columns > len(attributes):
            for i in range(len(attributes), max_columns):
                attributes.append(f"unknown_{i + 1}")
            logger.info(f"Extended attribute list to {len(attributes)} names")

        # Create a list of lists with consistent length
        padded_rows = []
        for row in data_rows:
            # Pad or truncate rows to match attribute list
            if len(row) < len(attributes):
                padded_row = row + [pd.NA] * (len(attributes) - len(row))
            else:
                padded_row = row[:len(attributes)]
            padded_rows.append(padded_row)

        # Create DataFrame with the consistent rows
        df = pd.DataFrame(padded_rows, columns=attributes)

        # Convert numeric columns
        numeric_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Clean categorical columns
        categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
                            'class']
        for col in categorical_cols:
            if col in df.columns:
                # Strip spaces and lowercase
                df[col] = df[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

        logger.info(f"Successfully loaded CKD dataset with shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
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
    raw_data_path = "../../data/raw/chronic_kidney_disease.arff"
    processed_data_path = "../../data/processed/ckd_processed.csv"

    # Load data
    ckd_df = load_ckd_data(raw_data_path)

    # Display basic information about the dataset
    logger.info(f"Dataset information:")
    logger.info(f"Number of samples: {len(ckd_df)}")
    logger.info(f"Number of features: {len(ckd_df.columns)}")
    logger.info(f"Column names: {ckd_df.columns.tolist()}")

    # Display first few rows
    logger.info("First few rows of the dataset:")
    logger.info(ckd_df.head())

    # Save processed data
    save_processed_data(ckd_df, processed_data_path)