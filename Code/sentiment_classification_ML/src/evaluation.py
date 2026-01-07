# coding: utf-8
'''
filename: evaluation.py
function: 模型评估相关函数
'''

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def calculate_average_reports(reports, labels):
    '''
    计算K折交叉验证的平均报告
    
    Args:
        reports: 包含每一折评估结果的列表
        labels: 需要计算平均值的标签列表
    
    Returns:
        avg_report: 平均分类报告
        avg_acc: 平均准确率
        avg_roc_auc: 平均ROC-AUC值
    '''
    # 准确率
    acc = []
    # auc值
    roc_auc = []
    # 初始化用于保存累计结果的字典
    total = {str(label): {'precision': 0, 'recall': 0, 'f1-score': 0} for label in labels}

    # 遍历报告列表，累加每个标签的 'precision'、'recall' 和 'f1-score'
    for report in reports:
        acc.append(report['accuracy'])
        roc_auc.append(report['roc_auc'])
        for label in labels:
            label = str(label)  # 确保标签是字符串
            if label in report:
                total[label]['precision'] += report[label]['precision']
                total[label]['recall'] += report[label]['recall']
                total[label]['f1-score'] += report[label]['f1-score']

    # 计算平均值
    num_reports = len(reports)
    avg_report = {str(label): {} for label in labels}
    for label in labels:
        label = str(label)  # 确保标签是字符串
        avg_report[label]['precision'] = total[label]['precision'] / num_reports
        avg_report[label]['recall'] = total[label]['recall'] / num_reports
        avg_report[label]['f1-score'] = total[label]['f1-score'] / num_reports

    # 计算平均acc
    avg_acc = np.mean(acc)

    # 计算平均roc_auc
    avg_roc_auc = np.mean(roc_auc)

    return avg_report, avg_acc, avg_roc_auc

def print_classification_results(y_true, y_pred, y_pred_proba=None):
    '''
    打印分类结果
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_proba: 预测概率，用于计算ROC-AUC
    
    Returns:
        report_dict: 分类报告字典
    '''
    # 生成分类报告
    report_text = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    print("分类报告:")
    print(report_text)
    
    # 如果提供了预测概率，计算ROC-AUC
    if y_pred_proba is not None:
        # 检查是二分类还是多分类
        num_classes = len(np.unique(y_true))
        if num_classes == 2:
            # 二分类情况
            try:
                # 尝试获取正类的概率
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                print(f"ROC-AUC值: {roc_auc:.4f}")
                report_dict['roc_auc'] = roc_auc
            except IndexError:
                # 如果y_pred_proba是一维数组
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                print(f"ROC-AUC值: {roc_auc:.4f}")
                report_dict['roc_auc'] = roc_auc
        else:
            # 多分类情况
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
            print(f"多分类ROC-AUC值: {roc_auc:.4f}")
            report_dict['roc_auc'] = roc_auc
    
    return report_dict

def summarize_k_fold_results(reports, labels):
    '''
    总结K折交叉验证的结果
    
    Args:
        reports: 包含每一折评估结果的列表
        labels: 需要计算平均值的标签列表
    
    Returns:
        None
    '''
    avg_report, avg_acc, avg_roc_auc = calculate_average_reports(reports, labels)
    
    print("\n======= K折交叉验证平均结果 =======")
    print(f"平均准确率: {avg_acc:.4f}")
    print(f"平均ROC-AUC值: {avg_roc_auc:.4f}")
    print("\n各类别平均性能指标:")
    
    for label in labels:
        label_str = str(label)
        print(f"类别 {label}:")
        print(f"  Precision: {avg_report[label_str]['precision']:.4f}")
        print(f"  Recall: {avg_report[label_str]['recall']:.4f}")
        print(f"  F1-Score: {avg_report[label_str]['f1-score']:.4f}")
    
    return avg_report, avg_acc, avg_roc_auc