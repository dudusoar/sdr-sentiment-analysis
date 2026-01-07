# coding: utf-8
'''
filename: utils.py
function: General utility functions
'''

import os
import pandas as pd
import re
from config import CONTRACTIONS

def ensure_directory(directory):
    """Ensure directory exists, create if not exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")

def save_excel(df, filepath):
    """Save DataFrame to Excel file"""
    directory = os.path.dirname(filepath)
    ensure_directory(directory)
    df.to_excel(filepath, index=False)
    print(f"File saved: {filepath}")

def load_excel(filepath):
    """Load Excel file to DataFrame"""
    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return None
    return pd.read_excel(filepath)

def word_replace(text):
    """Replace contractions in text with their expanded forms"""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

def is_digit(word):
    """Check if a word is a digit"""
    return bool(re.match(r'\d+', word))