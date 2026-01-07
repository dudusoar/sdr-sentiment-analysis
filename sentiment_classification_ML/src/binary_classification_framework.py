# coding: utf-8
'''
filename: binary_classification_framework.py
function: Binary classification framework
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.vectorizers import Word2VecVectorizer
from src.data_manager import load_data
from config import RANDOM_SEED
import os

class BinaryClassificationFramework:
    """Binary classification framework"""
    
    def __init__(self, test_size=0.3, random_state=RANDOM_SEED):
        self.test_size = test_size
        self.random_state = random_state
        
    def create_ovo_datasets(self, data_dir='results/dataset'):
        """Generate datasets for OVO (One-vs-One) strategy"""
        print("Generating OVO strategy datasets...")
        
        # OVO dataset file list (One-vs-One strategy)
        ovo_files = [
            'data_01.xlsx',  # Label 0 vs Label 1
            'data_02.xlsx',  # Label 0 vs Label 2  
            'data_12.xlsx'   # Label 1 vs Label 2
        ]
        
        datasets = []
        dataset_names = []
        
        for file in ovo_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                # Version with low-frequency words removed
                data_r = load_data(file_path, remove_punctuation=False, remove_low_frequency=True)
                # Version with low-frequency words kept
                data_k = load_data(file_path, remove_punctuation=False, remove_low_frequency=False)
                
                if data_r is not None and data_k is not None:
                    datasets.extend([data_r, data_k])
                    base_name = file.replace('.xlsx', '')
                    dataset_names.extend([f'{base_name}_r', f'{base_name}_k'])
                    print(f"Loading dataset: {file}")
                else:
                    print(f"Warning: Cannot load dataset {file}")
            else:
                print(f"Warning: Dataset file does not exist {file_path}")
        
        return datasets, dataset_names
    
    def create_ovr_datasets(self, data_dir='results/dataset'):
        """Generate datasets for OVR (One-vs-Rest) strategy"""
        print("Generating OVR strategy datasets...")
        
        # OVR dataset file list (One-vs-Rest strategy)
        ovr_files = [
            'data_0.xlsx',   # Label 0 vs Others (labels 1,2 converted to label 9)
            'data_1.xlsx',   # Label 1 vs Others (labels 0,2 converted to label 9)
            'data_2.xlsx'    # Label 2 vs Others (labels 0,1 converted to label 9)
        ]
        
        datasets = []
        dataset_names = []
        
        for file in ovr_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                # Version with low-frequency words removed
                data_r = load_data(file_path, remove_punctuation=False, remove_low_frequency=True)
                # Version with low-frequency words kept
                data_k = load_data(file_path, remove_punctuation=False, remove_low_frequency=False)
                
                if data_r is not None and data_k is not None:
                    datasets.extend([data_r, data_k])
                    base_name = file.replace('.xlsx', '')
                    dataset_names.extend([f'{base_name}_r', f'{base_name}_k'])
                    print(f"Loading dataset: {file}")
                else:
                    print(f"Warning: Cannot load dataset {file}")
            else:
                print(f"Warning: Dataset file does not exist {file_path}")
        
        return datasets, dataset_names
    
    def single_train(self, comments, oversampling, clf, vectorizer, test_size):
        """
        Single training function

        Args:
            comments: Dataset DataFrame
            oversampling: Whether to use oversampling
            clf: Classifier
            vectorizer: Feature vectorizer
            test_size: Test set ratio

        Returns:
            roc_auc: ROC-AUC value
            report_text: Classification report text
        """
        data = comments.copy()
        
        # Process string format required by TF-IDF vectorizer
        if isinstance(vectorizer, TfidfVectorizer):
            # Convert word list to string
            if 'word_list' in data.columns:
                data['word_list'] = data['word_list'].apply(' '.join)
            elif 'f_word_list' in data.columns:
                data['word_list'] = data['f_word_list'].apply(' '.join)
        
        # Prepare features and labels
        X = data['word_list'] if 'word_list' in data.columns else data['f_word_list']
        y = data['label1']
        
        # Split training and test sets
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Feature vectorization
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        
        # Oversampling processing - SMOTE needs to be performed after vectorization
        if oversampling:
            print('This training uses oversampling')
            print('Sample distribution before oversampling')
            print(y_train.value_counts())
            print('*' * 20)
            
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            print('Sample distribution after oversampling')
            print(pd.Series(y_train).value_counts())
            print('*' * 20)
        else:
            print('This training does not use oversampling')
            print('Sample distribution')
            print(y_train.value_counts())
            print('*' * 20)
        
        # Train model
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # 计算ROC-AUC (二分类情况下取第二列概率)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # 生成分类报告
        report_text = classification_report(y_test, y_pred)
        
        return roc_auc, report_text
    
    def binary_train(self, datasets, datasets_names, oversampling, clf, vectorizer, test_size):
        """
        Binary classification training framework

        Args:
            datasets: Dataset list
            datasets_names: Dataset names list
            oversampling: Whether to use oversampling
            clf: Classifier
            vectorizer: Feature vectorizer
            test_size: Test set ratio

        Returns:
            roc_auc_values: ROC-AUC values list
            report_texts: Classification report texts list
        """
        roc_auc_values = []
        report_texts = []
        
        for i in range(len(datasets)):
            dataset = datasets[i]
            dataset_name = datasets_names[i]
            
            print(f"\n========== Training {dataset_name} ==========")
            
            # Single training
            roc_auc, report_text = self.single_train(
                dataset, oversampling, clf, vectorizer, test_size
            )
            
            # Print results
            print(f'ROC-AUC value for {dataset_name} is {roc_auc}')
            print(f'Classification result for {dataset_name} is')
            print(report_text)
            print('-' * 50)
            
            # Save results
            roc_auc_values.append(roc_auc)
            report_texts.append(report_text)
        
        return roc_auc_values, report_texts
    
    def test_model_comprehensive(self, model_configs, feature_configs, strategy='both'):
        """
        Comprehensive test model

        Args:
            model_configs: Model configuration list
            feature_configs: Feature configuration list
            strategy: Strategy selection ('ovo', 'ovr', 'both')

        Returns:
            all_results: All test results
        """
        all_results = {}
        
        # Load datasets
        if strategy in ['ovo', 'both']:
            ovo_datasets, ovo_names = self.create_ovo_datasets()
            all_results['ovo'] = {}
        
        if strategy in ['ovr', 'both']:
            ovr_datasets, ovr_names = self.create_ovr_datasets()
            all_results['ovr'] = {}
        
        # 遍历所有模型配置
        for model_name, model_info in model_configs.items():
            print(f"\n{'='*60}")
            print(f"测试模型: {model_name}")
            print(f"{'='*60}")
            
            clf = model_info['model']
            oversampling = model_info.get('oversampling', False)
            
            # 遍历所有特征配置
            for feature_name, feature_info in feature_configs.items():
                print(f"\n{'-'*40}")
                print(f"特征配置: {feature_name}")
                print(f"{'-'*40}")
                
                vectorizer = feature_info['vectorizer']
                
                # 测试OVO策略
                if strategy in ['ovo', 'both']:
                    print(f"\n## OVO策略 - {model_name} + {feature_name}")
                    
                    ovo_roc_aucs, ovo_reports = self.binary_train(
                        ovo_datasets, ovo_names, oversampling, clf, vectorizer, self.test_size
                    )
                    
                    result_key = f"{model_name}_{feature_name}"
                    all_results['ovo'][result_key] = {
                        'roc_aucs': ovo_roc_aucs,
                        'reports': ovo_reports,
                        'dataset_names': ovo_names
                    }
                
                # 测试OVR策略
                if strategy in ['ovr', 'both']:
                    print(f"\n## OVR策略 - {model_name} + {feature_name}")
                    
                    ovr_roc_aucs, ovr_reports = self.binary_train(
                        ovr_datasets, ovr_names, oversampling, clf, vectorizer, self.test_size
                    )
                    
                    result_key = f"{model_name}_{feature_name}"
                    all_results['ovr'][result_key] = {
                        'roc_aucs': ovr_roc_aucs,
                        'reports': ovr_reports,
                        'dataset_names': ovr_names
                    }
        
        return all_results
    
    def find_best_configurations(self, all_results):
        """找出最优配置"""
        best_configs = {}
        
        for strategy in all_results:
            print(f"\n{'='*50}")
            print(f"{strategy.upper()}策略最优结果:")
            print(f"{'='*50}")
            
            best_roc_auc = 0
            best_config = None
            best_dataset = None
            
            for config_name, results in all_results[strategy].items():
                roc_aucs = results['roc_aucs']
                dataset_names = results['dataset_names']
                
                for i, roc_auc in enumerate(roc_aucs):
                    if roc_auc > best_roc_auc:
                        best_roc_auc = roc_auc
                        best_config = config_name
                        best_dataset = dataset_names[i]
            
            best_configs[strategy] = {
                'config': best_config,
                'dataset': best_dataset,
                'roc_auc': best_roc_auc
            }
            
            print(f"最优配置: {best_config}")
            print(f"最优数据集: {best_dataset}")
            print(f"最优ROC-AUC: {best_roc_auc:.4f}")
        
        return best_configs

def run_comprehensive_binary_test(word2vec_model=None):
    """运行完整的二分类测试"""
    
    print("开始运行完整的二分类测试框架...")
    
    # 初始化框架
    framework = BinaryClassificationFramework()
    
    # 定义模型配置 - 还原测试代码中的所有模型
    model_configs = {
        'MultinomialNB_no_sampling': {
            'model': MultinomialNB(),
            'oversampling': False
        },
        'MultinomialNB_with_sampling': {
            'model': MultinomialNB(),
            'oversampling': True
        },
        'SVM_no_sampling': {
            'model': SVC(probability=True, kernel='rbf'),
            'oversampling': False
        },
        'SVM_with_sampling': {
            'model': SVC(probability=True, kernel='rbf'),
            'oversampling': True
        },
        'RandomForest_10_no_sampling': {
            'model': RandomForestClassifier(n_estimators=10, random_state=42),
            'oversampling': False
        },
        'RandomForest_10_with_sampling': {
            'model': RandomForestClassifier(n_estimators=10, random_state=42),
            'oversampling': True
        },
        'RandomForest_100_no_sampling': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'oversampling': False
        },
        'RandomForest_100_with_sampling': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'oversampling': True
        },
        'DecisionTree_no_sampling': {
            'model': DecisionTreeClassifier(random_state=42),
            'oversampling': False
        },
        'DecisionTree_with_sampling': {
            'model': DecisionTreeClassifier(random_state=42),
            'oversampling': True
        }
    }
    
    # 定义特征配置 - 还原测试代码中的所有特征提取方法
    feature_configs = {
        'TF-IDF_1gram': {
            'vectorizer': TfidfVectorizer(ngram_range=(1,1))
        },
        'TF-IDF_2gram': {
            'vectorizer': TfidfVectorizer(ngram_range=(1,2))
        },
        'TF-IDF_3gram': {
            'vectorizer': TfidfVectorizer(ngram_range=(1,3))
        }
    }
    
    # 如果提供了Word2Vec模型，添加Word2Vec特征配置
    if word2vec_model is not None:
        feature_configs['Word2Vec_avg'] = {
            'vectorizer': Word2VecVectorizer(word2vec_model, bow='avg', shift_to_positive=False)
        }
        feature_configs['Word2Vec_avg_positive'] = {
            'vectorizer': Word2VecVectorizer(word2vec_model, bow='avg', shift_to_positive=True)
        }
    
    # 运行综合测试
    all_results = framework.test_model_comprehensive(
        model_configs, feature_configs, strategy='both'
    )
    
    # 找出最优配置
    best_configs = framework.find_best_configurations(all_results)
    
    return all_results, best_configs

def run_specific_test(model_name='MultinomialNB', feature_type='TF-IDF', strategy='ovo', 
                     oversampling=False, ngram_range=(1,1), word2vec_model=None):
    """运行特定配置的测试"""
    
    print(f"运行特定测试: {model_name} + {feature_type} + {strategy}")
    
    # 初始化框架
    framework = BinaryClassificationFramework()
    
    # 配置模型
    if model_name == 'MultinomialNB':
        clf = MultinomialNB()
    elif model_name == 'SVM':
        clf = SVC(probability=True, kernel='rbf')
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'DecisionTree':
        clf = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 配置特征提取器
    if feature_type == 'TF-IDF':
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    elif feature_type == 'Word2Vec':
        if word2vec_model is None:
            raise ValueError("使用Word2Vec特征需要提供word2vec_model参数")
        vectorizer = Word2VecVectorizer(word2vec_model, bow='avg', shift_to_positive=False)
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")
    
    # 加载数据集
    if strategy == 'ovo':
        datasets, dataset_names = framework.create_ovo_datasets()
    elif strategy == 'ovr':
        datasets, dataset_names = framework.create_ovr_datasets()
    else:
        raise ValueError(f"不支持的策略: {strategy}")
    
    # 运行测试
    roc_auc_values, report_texts = framework.binary_train(
        datasets, dataset_names, oversampling, clf, vectorizer, framework.test_size
    )
    
    return {
        'roc_aucs': roc_auc_values,
        'reports': report_texts,
        'dataset_names': dataset_names
    } 