# coding: utf-8
'''
filename: data_manager.py
function: Data loading, preprocessing, and splitting
'''

import pandas as pd
import os
from src.preprocessing import first_preprocessing, second_preprocessing, remove_low_frequency_words
from config import DATASET_DIR, SELECTED_COMMENTS_FILE, COMBINED_COMMENTS_FILE
from utils import save_excel, load_excel, ensure_directory

def load_data(file_name=SELECTED_COMMENTS_FILE, remove_punctuation=True, remove_low_frequency=True, low_frequency_words=None):
    '''
    Load data and perform preprocessing

    Args:
        file_name: Data file path
        remove_punctuation: Whether to remove punctuation
        remove_low_frequency: Whether to remove low-frequency words
        low_frequency_words: Low-frequency words list

    Returns:
        comments: Processed DataFrame
    '''
    comments = load_excel(file_name)
    if comments is None:
        return None
    
    # Lowercase, tokenization, remove useless punctuation, remove numbers, remove stopwords
    comments['f_text'] = comments['pure_text'].apply(first_preprocessing)
    # Version keeping stopwords
    comments['k_text'] = comments['pure_text'].apply(first_preprocessing, filtering=False)

    # Determine whether to remove punctuation based on parameter
    comments['f_word_list'] = comments['f_text'].apply(
        second_preprocessing, 
        remove_punctuation=remove_punctuation, 
        lemmatization=True
    )
    comments['k_word_list'] = comments['k_text'].apply(
        second_preprocessing, 
        remove_punctuation=remove_punctuation, 
        lemmatization=True
    )
    
    # Determine whether to remove low-frequency words based on parameter
    if remove_low_frequency and low_frequency_words is not None:
        comments['f_word_list'] = comments['f_word_list'].apply(
            remove_low_frequency_words, 
            low_frequency_words=low_frequency_words
        )
        comments['k_word_list'] = comments['k_word_list'].apply(
            remove_low_frequency_words, 
            low_frequency_words=low_frequency_words
        )
    
    # Filter out empty lists
    mask = comments['f_word_list'].apply(lambda x: x != [])
    comments = comments[mask]
    comments = comments.reset_index(drop=True)  # Reset index

    return comments

def filter_selected_comments(input_file, output_file=SELECTED_COMMENTS_FILE):
    '''
    Filter out data with labels 0, 1, 2

    Args:
        input_file: Input file path
        output_file: Output file path

    Returns:
        selected_comments: Filtered DataFrame
    '''
    comments = load_excel(input_file)
    if comments is None:
        return None
    
    index = comments['label1'].isin([0, 1, 2])
    selected_comments = comments[index]
    save_excel(selected_comments, output_file)
    
    return selected_comments

def create_ovo_datasets(data):
    '''
    Generate datasets for One-vs-One (OVO) strategy

    Args:
        data: DataFrame containing labels

    Returns:
        None
    '''
    ensure_directory(DATASET_DIR)
    
    index_01 = data['label1'].isin([0, 1])
    index_02 = data['label1'].isin([0, 2])
    index_12 = data['label1'].isin([1, 2])

    data_01 = data[index_01].reset_index(drop=True)
    data_02 = data[index_02].reset_index(drop=True)
    data_12 = data[index_12].reset_index(drop=True)

    # Save files
    save_excel(data_01, os.path.join(DATASET_DIR, 'data_01.xlsx'))
    save_excel(data_02, os.path.join(DATASET_DIR, 'data_02.xlsx'))
    save_excel(data_12, os.path.join(DATASET_DIR, 'data_12.xlsx'))

def create_ovr_datasets(data):
    '''
    Generate datasets for One-vs-Rest (OVR) strategy

    Args:
        data: DataFrame containing labels

    Returns:
        None
    '''
    ensure_directory(DATASET_DIR)
    
    # Keep original labels, mark other labels as 9
    data_0 = data.copy()
    data_0['label1'] = data_0['label1'].apply(lambda x: 0 if x == 0 else 9)

    data_1 = data.copy()
    data_1['label1'] = data_1['label1'].apply(lambda x: 1 if x == 1 else 9)

    data_2 = data.copy()
    data_2['label1'] = data_2['label1'].apply(lambda x: 2 if x == 2 else 9)

    # Save files
    save_excel(data_0, os.path.join(DATASET_DIR, 'data_0.xlsx'))
    save_excel(data_1, os.path.join(DATASET_DIR, 'data_1.xlsx'))
    save_excel(data_2, os.path.join(DATASET_DIR, 'data_2.xlsx'))

def create_datasets():
    '''
    Main function: Create all datasets
    '''
    # Filter data
    selected_comments = filter_selected_comments(COMBINED_COMMENTS_FILE)
    if selected_comments is None:
        print("Cannot load original data file")
        return
    
    # Create OVO and OVR datasets
    create_ovo_datasets(selected_comments)
    create_ovr_datasets(selected_comments)
    print("All datasets created")

def load_training_datasets(type='ovo', remove_punctuation=False, remove_low_frequency=True, low_frequency_words=None):
    '''
    Load training datasets

    Args:
        type: 'ovo' or 'ovr'
        remove_punctuation: Whether to remove punctuation
        remove_low_frequency: Whether to remove low-frequency words
        low_frequency_words: Low-frequency words list

    Returns:
        datasets: Dataset list
        datasets_names: Dataset names list
    '''
    if type == 'ovo':
        # OVO datasets
        data_01_r = load_data(os.path.join(DATASET_DIR, 'data_01.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_01_k = load_data(os.path.join(DATASET_DIR, 'data_01.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        data_02_r = load_data(os.path.join(DATASET_DIR, 'data_02.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_02_k = load_data(os.path.join(DATASET_DIR, 'data_02.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        data_12_r = load_data(os.path.join(DATASET_DIR, 'data_12.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_12_k = load_data(os.path.join(DATASET_DIR, 'data_12.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        
        datasets = [data_01_r, data_01_k, data_02_r, data_02_k, data_12_r, data_12_k]
        datasets_names = ['data_01_r', 'data_01_k', 'data_02_r', 'data_02_k', 'data_12_r', 'data_12_k']
    
    elif type == 'ovr':
        # OVR datasets
        data_0_r = load_data(os.path.join(DATASET_DIR, 'data_0.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_0_k = load_data(os.path.join(DATASET_DIR, 'data_0.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        data_1_r = load_data(os.path.join(DATASET_DIR, 'data_1.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_1_k = load_data(os.path.join(DATASET_DIR, 'data_1.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        data_2_r = load_data(os.path.join(DATASET_DIR, 'data_2.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_2_k = load_data(os.path.join(DATASET_DIR, 'data_2.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        
        datasets = [data_0_r, data_0_k, data_1_r, data_1_k, data_2_r, data_2_k]
        datasets_names = ['data_0_r', 'data_0_k', 'data_1_r', 'data_1_k', 'data_2_r', 'data_2_k']
    
    else:
        # Multi-class datasets
        data_r = load_data(SELECTED_COMMENTS_FILE, 
                          remove_punctuation=remove_punctuation, 
                          remove_low_frequency=remove_low_frequency,
                          low_frequency_words=low_frequency_words)
        data_k = load_data(SELECTED_COMMENTS_FILE, 
                          remove_punctuation=remove_punctuation, 
                          remove_low_frequency=False,
                          low_frequency_words=low_frequency_words)
        
        datasets = [data_r, data_k]
        datasets_names = ['data_r', 'data_k']
    
    return datasets, datasets_names

if __name__ == '__main__':
    create_datasets()