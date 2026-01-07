#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Initialize NLTK data download script
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download necessary NLTK data"""
    datasets = ['stopwords', 'punkt_tab', 'punkt', 'wordnet']
    
    for dataset in datasets:
        print(f"Downloading NLTK {dataset} data...")
        try:
            nltk.download(dataset, quiet=True)
            print(f"NLTK {dataset} data download completed!")
        except Exception as e:
            print(f"Download {dataset} failed: {e}")
            # Continue downloading other datasets
            continue
    
    print("All NLTK data download completed!")
    return True

if __name__ == "__main__":
    download_nltk_data() 