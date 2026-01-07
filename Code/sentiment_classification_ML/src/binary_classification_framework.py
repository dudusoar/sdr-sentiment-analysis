# coding: utf-8
'''
filename: binary_classification_framework.py
function: 二分类框架 - 完整还原测试代码中的二分类测试流程
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
    """二分类框架 - 完整还原测试代码功能"""
    
    def __init__(self, test_size=0.3, random_state=RANDOM_SEED):
        self.test_size = test_size
        self.random_state = random_state
        
    def create_ovo_datasets(self, data_dir='results/dataset'):
        """生成OVO策略的数据集"""
        print("正在生成OVO策略数据集...")
        
        # OVO数据集文件列表 (一对一策略)
        ovo_files = [
            'data_01.xlsx',  # 标签0 vs 标签1
            'data_02.xlsx',  # 标签0 vs 标签2  
            'data_12.xlsx'   # 标签1 vs 标签2
        ]
        
        datasets = []
        dataset_names = []
        
        for file in ovo_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                # 去除低频词版本
                data_r = load_data(file_path, remove_punctuation=False, remove_low_frequency=True)
                # 保留低频词版本
                data_k = load_data(file_path, remove_punctuation=False, remove_low_frequency=False)
                
                if data_r is not None and data_k is not None:
                    datasets.extend([data_r, data_k])
                    base_name = file.replace('.xlsx', '')
                    dataset_names.extend([f'{base_name}_r', f'{base_name}_k'])
                    print(f"加载数据集: {file}")
                else:
                    print(f"警告: 无法加载数据集 {file}")
            else:
                print(f"警告: 数据集文件不存在 {file_path}")
        
        return datasets, dataset_names
    
    def create_ovr_datasets(self, data_dir='results/dataset'):
        """生成OVR策略的数据集"""
        print("正在生成OVR策略数据集...")
        
        # OVR数据集文件列表 (一对多策略)
        ovr_files = [
            'data_0.xlsx',   # 标签0 vs 其他(标签1,2转为标签9)
            'data_1.xlsx',   # 标签1 vs 其他(标签0,2转为标签9)
            'data_2.xlsx'    # 标签2 vs 其他(标签0,1转为标签9)
        ]
        
        datasets = []
        dataset_names = []
        
        for file in ovr_files:
            file_path = os.path.join(data_dir, file)
            if os.path.exists(file_path):
                # 去除低频词版本
                data_r = load_data(file_path, remove_punctuation=False, remove_low_frequency=True)
                # 保留低频词版本
                data_k = load_data(file_path, remove_punctuation=False, remove_low_frequency=False)
                
                if data_r is not None and data_k is not None:
                    datasets.extend([data_r, data_k])
                    base_name = file.replace('.xlsx', '')
                    dataset_names.extend([f'{base_name}_r', f'{base_name}_k'])
                    print(f"加载数据集: {file}")
                else:
                    print(f"警告: 无法加载数据集 {file}")
            else:
                print(f"警告: 数据集文件不存在 {file_path}")
        
        return datasets, dataset_names
    
    def single_train(self, comments, oversampling, clf, vectorizer, test_size):
        """
        单次训练函数 - 完全还原测试代码逻辑
        
        Args:
            comments: 数据集DataFrame
            oversampling: 是否采用过采样
            clf: 分类器
            vectorizer: 特征向量化器
            test_size: 测试集比例
        
        Returns:
            roc_auc: ROC-AUC值
            report_text: 分类报告文本
        """
        data = comments.copy()
        
        # 处理TF-IDF向量化器需要的字符串格式
        if isinstance(vectorizer, TfidfVectorizer):
            # 将词列表转换为字符串
            if 'word_list' in data.columns:
                data['word_list'] = data['word_list'].apply(' '.join)
            elif 'f_word_list' in data.columns:
                data['word_list'] = data['f_word_list'].apply(' '.join)
        
        # 准备特征和标签
        X = data['word_list'] if 'word_list' in data.columns else data['f_word_list']
        y = data['label1']
        
        # 分割训练集和测试集
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # 特征向量化
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        
        # 过采样处理 - SMOTE需要在向量化后进行
        if oversampling:
            print('此次训练【采用】过采样')
            print('过采样【前】的样本分布')
            print(y_train.value_counts())
            print('*' * 20)
            
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            print('过采样【后】的样本分布')
            print(pd.Series(y_train).value_counts())
            print('*' * 20)
        else:
            print('此次训练【没有采用】过采样')
            print('样本分布')
            print(y_train.value_counts())
            print('*' * 20)
        
        # 训练模型
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
        二分类训练框架 - 完全还原测试代码逻辑
        
        Args:
            datasets: 数据集列表
            datasets_names: 数据集名称列表
            oversampling: 是否采用过采样
            clf: 分类器
            vectorizer: 特征向量化器
            test_size: 测试集比例
        
        Returns:
            roc_auc_values: ROC-AUC值列表
            report_texts: 分类报告文本列表
        """
        roc_auc_values = []
        report_texts = []
        
        for i in range(len(datasets)):
            dataset = datasets[i]
            dataset_name = datasets_names[i]
            
            print(f"\n========== 训练 {dataset_name} ==========")
            
            # 单次训练
            roc_auc, report_text = self.single_train(
                dataset, oversampling, clf, vectorizer, test_size
            )
            
            # 打印结果
            print(f'{dataset_name}的roc_auc值是{roc_auc}')
            print(f'{dataset_name}的分类结果是')
            print(report_text)
            print('-' * 50)
            
            # 保存结果
            roc_auc_values.append(roc_auc)
            report_texts.append(report_text)
        
        return roc_auc_values, report_texts
    
    def test_model_comprehensive(self, model_configs, feature_configs, strategy='both'):
        """
        综合测试模型 - 完整还原测试代码的测试流程
        
        Args:
            model_configs: 模型配置列表
            feature_configs: 特征配置列表
            strategy: 策略选择 ('ovo', 'ovr', 'both')
        
        Returns:
            all_results: 所有测试结果
        """
        all_results = {}
        
        # 加载数据集
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
    """运行完整的二分类测试 - 完全还原测试代码功能"""
    
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