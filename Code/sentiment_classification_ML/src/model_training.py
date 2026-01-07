# coding: utf-8
'''
filename: model_training.py
function: 模型训练相关函数
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
    K折交叉验证训练并返回评估结果
    
    Args:
        X: 特征数据
        y: 标签数据
        clf: 分类器
        k: 折数
        vectorizer: 特征向量化器
    
    Returns:
        reports: 包含每一折评估结果的列表
    '''
    kf = KFold(n_splits=k)
    reports = []
    fold = 0
    
    for train_index, test_index in kf.split(X):
        fold += 1
        print(f"\n====== 第{fold}折训练 ======")
        
        # 分割训练集和测试集
        X_train_raw, X_test_raw = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 向量化处理
        if vectorizer is not None:
            # 处理TF-IDF向量化器需要的字符串格式
            if isinstance(vectorizer, TfidfVectorizer) and isinstance(X_train_raw[0], list):
                X_train_raw = [' '.join(x) for x in X_train_raw]
                X_test_raw = [' '.join(x) for x in X_test_raw]
            
            # 在训练集上拟合向量化器
            X_train = vectorizer.fit_transform(X_train_raw)
            # 将向量化器应用到测试集
            X_test = vectorizer.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 测试模型
        y_pred = clf.predict(X_test)
        
        # 获取预测概率
        if hasattr(clf, 'predict_proba'):
            y_pred_proba = clf.predict_proba(X_test)
            
            # 计算ROC-AUC值
            num_classes = len(np.unique(y))
            if num_classes == 2:
                # 二分类
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                # 多分类
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
            
            print(f'第{fold}折训练的ROC-AUC值: {roc_auc:.4f}')
        else:
            y_pred_proba = None
            roc_auc = 0
            print("警告: 分类器不支持predict_proba方法，无法计算ROC-AUC")
        
        # 生成分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        report_text = classification_report(y_test, y_pred)
        print(f'第{fold}折训练结果:')
        print(report_text)
        
        # 将ROC-AUC添加到报告中
        report['roc_auc'] = roc_auc
        reports.append(report)
    
    return reports

def single_train(comments, clf, vectorizer=None, test_size=TEST_SIZE, oversampling=False):
    '''
    单次训练模型
    
    Args:
        comments: 数据DataFrame
        clf: 分类器
        vectorizer: 特征向量化器
        test_size: 测试集比例
        oversampling: 是否使用过采样
    
    Returns:
        roc_auc: ROC-AUC值
        report_dict: 分类报告字典
    '''
    data = comments.copy()
    
    # 处理TF-IDF向量化器需要的字符串格式
    if isinstance(vectorizer, TfidfVectorizer) and 'word_list' in data.columns and isinstance(data['word_list'].iloc[0], list):
        data['word_list'] = data['word_list'].apply(' '.join)

    X = data['word_list']
    y = data['label1']
    
    # 分割训练集和测试集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    # 向量化处理
    if vectorizer is not None:
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw
    
    # 过采样处理
    if oversampling:
        print('此次训练【采用】过采样')
        print('过采样【前】的样本分布:')
        print(pd.Series(y_train).value_counts())
        
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print('过采样【后】的样本分布:')
        print(pd.Series(y_train).value_counts())
    else:
        print('此次训练【没有采用】过采样')
        print('样本分布:')
        print(pd.Series(y_train).value_counts())
    
    # 训练模型
    clf.fit(X_train, y_train)
    
    # 测试模型
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