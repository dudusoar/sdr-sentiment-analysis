# coding: utf-8
'''
filename: main.py
function: Main program entry point
'''

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim.models import KeyedVectors

from config import create_directories, WORD_FREQ_DIR, DATASET_DIR, COMBINED_COMMENTS_FILE, SELECTED_COMMENTS_FILE, WORD2VEC_MODEL_PATH, DATA_DIR
from src.data_manager import filter_selected_comments, create_ovo_datasets, create_ovr_datasets, load_data, load_training_datasets
from src.preprocessing import first_preprocessing, second_preprocessing
from src.vectorizers import Word2VecVectorizer
from src.model_training import k_fold_train_report, single_train
from src.evaluation import print_classification_results
from src.visualization import analyze_word_frequency
from src.word2vec_downloader import download_word2vec_model, load_word2vec_model, fix_word2vec_encoding
from src.binary_classification_framework import run_specific_test, run_comprehensive_binary_test

def setup_word2vec():
    """Set up Word2Vec model"""
    print("Setting up Word2Vec model...")
    
    # Try to download the model
    success = download_word2vec_model(DATA_DIR, method='auto')
    
    if success:
        # Verify model can be loaded properly
        model = load_word2vec_model(DATA_DIR)
        if model is not None:
            print(f"[OK] Word2Vec model setup successful!")
            print(f"Vocabulary size: {len(model.key_to_index)}")
            print(f"Vector dimension: {model.vector_size}")
            return True
        else:
            print("[ERROR] Word2Vec model downloaded but failed to load")
            return False
    else:
        print("[ERROR] Word2Vec model setup failed")
        return False

def fix_word2vec():
    """Fix Word2Vec encoding issues"""
    print("[FIX] Starting to fix Word2Vec encoding issues...")
    return fix_word2vec_encoding(DATA_DIR)

def prepare_data():
    """Prepare data"""
    print("Starting data preparation...")

    # Create necessary directories
    create_directories()

    # Filter selected comments
    print("Filtering selected comments...")
    selected_comments = filter_selected_comments(COMBINED_COMMENTS_FILE, SELECTED_COMMENTS_FILE)

    if selected_comments is None:
        print("❌ Failed to filter selected comments")
        return

    # Create OVO dataset
    print("Creating OVO dataset...")
    create_ovo_datasets(selected_comments)

    # Create OVR dataset
    print("Creating OVR dataset...")
    create_ovr_datasets(selected_comments)

    print("[OK] Data preparation complete")

def analyze_data():
    """Analyze data"""
    print("Starting data analysis...")
    
    # Load data
    df = load_data(SELECTED_COMMENTS_FILE)
    
    # Preprocessing
    df['preprocessed'] = df['f_word_list'].apply(first_preprocessing)
    df['preprocessed'] = df['preprocessed'].apply(second_preprocessing)
    
    # Word frequency analysis
    word_freq_data = analyze_word_frequency(df, output_dir=WORD_FREQ_DIR)
    
    print("✅ Data analysis complete")

def train_models(model_name='all', classification_type='multi', ngram_range=1, use_kfold=False, use_word2vec=False):
    """Train models"""
    print(f"Starting model training: {model_name}")
    
    # Determine n-gram range
    if ngram_range == 1:
        ngram = (1, 1)
    elif ngram_range == 2:
        ngram = (1, 2)
    elif ngram_range == 3:
        ngram = (1, 3)
    else:
        ngram = (1, 1)
    
    # Load data
    if classification_type == 'multi':
        df = load_data(SELECTED_COMMENTS_FILE)
    else:
        # Binary classification requires specifying dataset
        df = load_data(SELECTED_COMMENTS_FILE)  # Can be extended to support different binary classification datasets
    
    # Preprocessing
    df['preprocessed'] = df['f_word_list'].apply(first_preprocessing)
    df['preprocessed'] = df['preprocessed'].apply(second_preprocessing)
    
    # Feature vectorization
    if use_word2vec:
        word2vec_model = load_word2vec_model('data')
        if word2vec_model is None:
            print("Word2Vec model failed to load, using TF-IDF")
            vectorizer = TfidfVectorizer(ngram_range=ngram)
        else:
            vectorizer = Word2VecVectorizer(word2vec_model)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram)
    
    # Model configuration
    models = {}
    if model_name in ['nb', 'all']:
        models['MultinomialNB'] = MultinomialNB()
    if model_name in ['svm', 'all']:
        models['SVM'] = SVC(probability=True)
    if model_name in ['rf', 'all']:
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    if model_name in ['dt', 'all']:
        models['DecisionTree'] = DecisionTreeClassifier(random_state=42)
    
    # Training and evaluation
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if use_kfold:
            # Prepare data for K-fold cross-validation
            X = df['preprocessed'].values
            y = df['label'].values
            
            # Process string format required for TF-IDF
            if isinstance(vectorizer, TfidfVectorizer):
                X = df['preprocessed'].apply(' '.join).values
            
            reports = k_fold_train_report(X, y, model, vectorizer=vectorizer)
        else:
            # Prepare data for single training
            data = df.copy()
            data['word_list'] = data['preprocessed']
            data['label1'] = data['label']
            
            result = single_train(data, model, vectorizer=vectorizer, oversampling=False)
    
    print("✅ Model training complete")

def binary_test(test_type='comprehensive', test_model=None, test_feature=None, test_strategy=None):
    """Run binary classification test"""
    print(f"Starting binary classification test: {test_type}")
    
    if test_type == 'comprehensive':
        # Run complete test
        run_comprehensive_binary_test()
    elif test_type == 'specific':
        # Run specific configuration test
        if not all([test_model, test_feature, test_strategy]):
            print("Specific test requires specifying model, feature, and strategy")
            return
        
        # Load Word2Vec model (if needed)
        word2vec_model = None
        if test_feature == 'Word2Vec':
            word2vec_model = load_word2vec_model('data')
            if word2vec_model is None:
                print("Word2Vec model loading failed, cannot proceed with test")
                return
        
        results = run_specific_test(
            model_name=test_model,
            feature_type=test_feature,
            strategy=test_strategy,
            oversampling=False,
            ngram_range=(1,1),
            word2vec_model=word2vec_model
        )
        
        print("Test results:")
        print(f"Average ROC-AUC: {sum(results['roc_aucs'])/len(results['roc_aucs']):.4f}")
        for name, auc in zip(results['dataset_names'], results['roc_aucs']):
            print(f"  {name}: {auc:.4f}")
    
    print("✅ Binary classification test completed")

def main():
    parser = argparse.ArgumentParser(description='YouTube delivery robot sentiment analysis')
    
    # Basic functions
    parser.add_argument('--prepare', action='store_true', help='Prepare data')
    parser.add_argument('--analyze', action='store_true', help='Analyze data')
    parser.add_argument('--train', action='store_true', help='Train models')
    
    # Word2Vec related
    parser.add_argument('--setup-word2vec', action='store_true', help='Set up Word2Vec model')
    parser.add_argument('--fix-word2vec', action='store_true', help='Fix Word2Vec encoding issues')
    
    # Model selection
    parser.add_argument('--model', choices=['nb', 'svm', 'rf', 'dt', 'all'], default='all', help='Select model')
    parser.add_argument('--type', choices=['multi', 'ovo', 'ovr', 'all'], default='multi', help='Classification type')
    parser.add_argument('--ngram', type=int, choices=[1, 2, 3], default=1, help='N-gram range')
    parser.add_argument('--kfold', action='store_true', help='Use K-fold cross-validation')
    parser.add_argument('--word2vec', action='store_true', help='Use Word2Vec features')
    
    # Binary classification test
    parser.add_argument('--binary-test', choices=['comprehensive', 'specific'], help='Run binary classification test')
    parser.add_argument('--test-model', choices=['MultinomialNB', 'SVM', 'RandomForest', 'DecisionTree'], help='Test model')
    parser.add_argument('--test-feature', choices=['TF-IDF', 'Word2Vec'], help='Test feature')
    parser.add_argument('--test-strategy', choices=['ovo', 'ovr'], help='Test strategy')
    
    args = parser.parse_args()
    
    # Execute functions
    if args.setup_word2vec:
        setup_word2vec()
    
    if args.fix_word2vec:
        fix_word2vec()
    
    if args.prepare:
        prepare_data()
    
    if args.analyze:
        analyze_data()
    
    if args.train:
        train_models(
            model_name=args.model,
            classification_type=args.type,
            ngram_range=args.ngram,
            use_kfold=args.kfold,
            use_word2vec=args.word2vec
        )
    
    if args.binary_test:
        binary_test(
            test_type=args.binary_test,
            test_model=args.test_model,
            test_feature=args.test_feature,
            test_strategy=args.test_strategy
        )

if __name__ == "__main__":
    main()