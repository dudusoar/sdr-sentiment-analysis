# sdr_clustering_analysis/src/data_loader.py
import pandas as pd
import os
from config import RAW_COMMENTS_FILE, TEXT_COLUMN, ID_COLUMN, MANUAL_SENTIMENT_COLUMN

def load_raw_comments(filepath=RAW_COMMENTS_FILE):
    """
    Load raw comment data from an Excel or CSV file.
    Ensure that the specified text column, ID column, and sentiment label column exist.
    """
    if not os.path.exists(filepath):
        print(f"Error: Raw data file not found: {filepath}")
        return None

    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            print(f"Error: Unsupported file format: {filepath}. Please use .xlsx or .csv.")
            return None
        print(f"Raw data loaded from {filepath}. Total comments: {len(df)}.")

        # Check whether required columns exist
        required_columns = [TEXT_COLUMN]
        if ID_COLUMN:  # ID column is optional but recommended
            required_columns.append(ID_COLUMN)
        if MANUAL_SENTIMENT_COLUMN:  # Manual sentiment label column is optional but used for later comparison
            required_columns.append(MANUAL_SENTIMENT_COLUMN)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: The following columns are missing from the raw data: {', '.join(missing_columns)}")
            print("Please check whether TEXT_COLUMN, ID_COLUMN, and MANUAL_SENTIMENT_COLUMN in config.py "
                  "match the column names in the Excel/CSV file.")
            print(f"Columns found in the file: {df.columns.tolist()}")
            return None

        # Remove rows where the text column is empty
        original_len = len(df)
        df.dropna(subset=[TEXT_COLUMN], inplace=True)
        df = df[df[TEXT_COLUMN].astype(str).str.strip() != '']
        if len(df) < original_len:
            print(f"Removed {original_len - len(df)} comments with empty text.")

        # Ensure the text column is of string type
        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

        # If an ID column is configured, ensure uniqueness; otherwise create a unique ID
        if ID_COLUMN and ID_COLUMN in df.columns:
            if df[ID_COLUMN].duplicated().any():
                print(f"Warning: Duplicate values found in the configured ID column '{ID_COLUMN}'. "
                      "A new unique ID column 'unique_comment_id' will be created.")
                df['unique_comment_id'] = range(len(df))
                df.rename(columns={ID_COLUMN: f"original_{ID_COLUMN}"}, inplace=True)  # Preserve original ID column
                # Update note: config itself will not change, but internally the new ID can be used
                # Alternatively, always rely on a fixed new ID column name in the returned DataFrame
                print("Please use 'unique_comment_id' as the unique identifier in subsequent analyses.")
            else:
                # If the original ID column is already unique, rename it to 'unique_comment_id'
                df.rename(columns={ID_COLUMN: 'unique_comment_id'}, inplace=True)

        elif not ID_COLUMN or ID_COLUMN not in df.columns:
            print(f"Note: No valid ID column configured ('{ID_COLUMN}'). "
                  "A new unique ID column 'unique_comment_id' will be created.")
            df['unique_comment_id'] = range(len(df))

        print(f"Data loading and initial processing completed. "
              f"Remaining valid comments: {len(df)}.")
        return df

    except Exception as e:
        print(f"Failed to load raw data file: {filepath}. Error: {e}")
        return None


if __name__ == '__main__':
    print("Testing data_loader.py...")
    # Assume combined_comments.xlsx is located in the DATA_DIR specified in config.py
    # and TEXT_COLUMN, etc., are correctly configured in config.py
    comments_df = load_raw_comments()

    if comments_df is not None:
        print(f"\nSuccessfully loaded {len(comments_df)} comments.")
        print("\nPreview of the first 5 comments (text, ID, sentiment label):")

        preview_cols = []
        # Use TEXT_COLUMN and MANUAL_SENTIMENT_COLUMN from config
        if TEXT_COLUMN in comments_df.columns:
            preview_cols.append(TEXT_COLUMN)

        # Prefer 'unique_comment_id' if it exists
        if 'unique_comment_id' in comments_df.columns:
            preview_cols.append('unique_comment_id')
        elif ID_COLUMN and ID_COLUMN in comments_df.columns:
            preview_cols.append(ID_COLUMN)

        if MANUAL_SENTIMENT_COLUMN and MANUAL_SENTIMENT_COLUMN in comments_df.columns:
            preview_cols.append(MANUAL_SENTIMENT_COLUMN)

        if preview_cols:
            print(comments_df[preview_cols].head())
        else:
            print("Unable to display preview. Please ensure column names in config.py are correct.")
            print(comments_df.head())

        print(f"\nDataFrame columns: {comments_df.columns.tolist()}")
    else:
        print("Data loading failed. Please check the error messages and file path/content.")
