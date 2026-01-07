# sdr_clustering_analysis/src/utils.py
import os
import pickle
import pandas as pd
import torch # For checking GPU availability

def save_pickle(data, filepath):
    """Save data to pickle file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save pickle file: {filepath}. Error: {e}")

def load_pickle(filepath):
    """Load data from pickle file."""
    if not os.path.exists(filepath):
        print(f"Pickle file not found: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        print(f"Failed to load pickle file: {filepath}. Error: {e}")
        return None

def save_csv(df, filepath, index=False):
    """Save DataFrame to CSV file."""
    try:
        df.to_csv(filepath, index=index, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
        print(f"CSV file saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save CSV file: {filepath}. Error: {e}")

def get_device_for_sbert():
    """Get appropriate device for SentenceTransformer (GPU or CPU)."""
    if torch.cuda.is_available():
        print("GPU detected, will use cuda device.")
        return 'cuda'
    else:
        print("No GPU detected, will use cpu device.")
        return 'cpu'

if __name__ == '__main__':
    # Simple test
    print("Testing utils.py...")
    
    # Test device detection
    device = get_device_for_sbert()
    print(f"SBERT will use device: {device}")

    # Test pickle save and load
    test_data_pkl = {"key": "value", "numbers": [1, 2, 3]}
    test_pkl_path = "test_data.pkl"
    save_pickle(test_data_pkl, test_pkl_path)
    loaded_data_pkl = load_pickle(test_pkl_path)
    if loaded_data_pkl == test_data_pkl:
        print("Pickle save and load test successful.")
    else:
        print("Pickle save and load test failed.")
    if os.path.exists(test_pkl_path):
        os.remove(test_pkl_path)

    # Test CSV save (simple example)
    test_df_csv = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    test_csv_path = "test_data.csv"
    save_csv(test_df_csv, test_csv_path)
    if os.path.exists(test_csv_path):
        print(f"CSV file {test_csv_path} save test successful (please check content manually).")
        os.remove(test_csv_path)
    else:
        print(f"CSV file {test_csv_path} save test failed.")

    print("utils.py test completed.")