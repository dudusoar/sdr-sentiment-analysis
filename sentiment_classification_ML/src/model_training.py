# coding: utf-8
'''
filename: model_training.py
function: Model training related functions
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.evaluation import print_classification_results, calculate_average_reports
from config import K_FOLDS, RANDOM_SEED, TEST_SIZE

def k_fold_train_report(X, y, clf, k=K_FOLDS, vectorizer=None):
    '''
    K-fold cross-validation training and return evaluation results

    Args:
        X: Feature data
        y: Label data
        clf: Classifier
        k: Number of folds
        vectorizer: Feature vectorizer

    Returns:
        reports: List containing evaluation results for each fold
    '''
    kf = KFold(n_splits=k)
    reports = []
    fold = 0
    
    for train_index, test_index in kf.split(X):
        fold += 1
        print(f"\n====== Fold {fold} training ======")
        
        # Split training and test sets
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Vectorization processing
        if vectorizer is not None:
            # Process string format required by TF-IDF vectorizer
            if isinstance(vectorizer, TfidfVectorizer) and isinstance(X_train_raw[0], list):
                X_train_raw = [' '.join(x) for x in X_train_raw]
                X_test_raw = [' '.join(x) for x in X_test_raw]
            
            # Fit vectorizer on training set
            X_train = vectorizer.fit_transform(X_train_raw)
            # Apply vectorizer to test set
            X_test = vectorizer.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw
        
        # Train model
        clf.fit(X_train, y_train)
        
        # Test model
        y_pred = clf.predict(X_test)
        
        # Get prediction probabilities
        if hasattr(clf, 'predict_proba'):
            y_pred_proba = clf.predict_proba(X_test)
            
            # Calculate ROC-AUC value
            num_classes = len(np.unique(y))
            if num_classes == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # Multi-class classification
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
            
            print(f'ROC-AUC value for fold {fold} training: {roc_auc:.4f}')
        else:
            y_pred_proba = None
            roc_auc = 0
            print("Warning: Classifier does not support predict_proba method, cannot calculate ROC-AUC")
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred)
        print(f'Fold {fold} training result:')
        print(report_text)
        
        # Add ROC-AUC to report
        report['roc_auc'] = roc_auc
        reports.append(report)
    
    return reports

def single_train(comments, clf, vectorizer=None, test_size=TEST_SIZE, oversampling=False):
    '''
    Single training model

    Args:
        comments: Data DataFrame
        clf: Classifier
        vectorizer: Feature vectorizer
        test_size: Test set ratio
        oversampling: Whether to use oversampling

    Returns:
        roc_auc: ROC-AUC value
        report_dict: Classification report dictionary
    '''
    data = comments.copy()
    
    # Process string format required by TF-IDF vectorizer
    if isinstance(vectorizer, TfidfVectorizer) and 'word_list' in data.columns and isinstance(data['word_list'].iloc[0], list):
        data['word_list'] = data['word_list'].apply(' '.join)

    X = data['word_list']
    y = data['label1']
    
    # Split training and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    # Vectorization processing
    if vectorizer is not None:
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw
    
    # Oversampling processing
    if oversampling:
        print('This training uses oversampling')
        print('Sample distribution before oversampling:')
        print(pd.Series(y_train).value_counts())
        
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print('Sample distribution after oversampling:')
        print(pd.Series(y_train).value_counts())
    else:
        print('This training does not use oversampling')
        print('Sample distribution:')
        print(pd.Series(y_train).value_counts())
    
    # Train model
    clf.fit(X_train, y_train)
    
    # Test model
    y_pred = clf.predict(X_test)
    
    # 获取预测概率
    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X_test)
    else:
        y_pred_proba = None
    
    # 打印分类结果
    report_dict = print_classification_results(y_test, y_pred, y_pred_proba)
    
    return report_dict

def binary_train(datasets, datasets_names, clf, vectorizer=None, test_size=TEST_SIZE, oversampling=False):
    '''
    在多个二分类数据集上训练模型
    
    Args:
        datasets: 数据集列表
        datasets_names: 数据集名称列表
        clf: 分类器
        vectorizer: 特征向量化器
        test_size: 测试集比例
        oversampling: 是否使用过采样
    
    Returns:
        results: 包含各个数据集评估结果的列表
        result_names: 结果名称列表
    '''
    # 保存结果
    results = []
    
    for i, dataset in enumerate(datasets):
        dataset_name = datasets_names[i]
        print(f"\n====== 在{dataset_name}上训练 ======")
        
        # 训练并评估模型
        report_dict = single_train(
            dataset, 
            clf=clf, 
            vectorizer=vectorizer, 
            test_size=test_size, 
            oversampling=oversampling
        )
        
        results.append(report_dict)
    
    return results, datasets_names