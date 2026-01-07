# coding: utf-8
'''
filename: evaluation.py
function: Model evaluation related functions
'''

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def calculate_average_reports(reports, labels):
    '''
    Calculate average reports for K-fold cross-validation

    Args:
        reports: List containing evaluation results for each fold
        labels: List of labels to calculate average for

    Returns:
        avg_report: Average classification report
        avg_acc: Average accuracy
        avg_roc_auc: Average ROC-AUC value
    '''
    # Accuracy
    acc = []
    # AUC value
    roc_auc = []
    # Initialize dictionary for storing cumulative results
    total = {str(label): {'precision': 0, 'recall': 0, 'f1-score': 0} for label in labels}

    # Iterate through report list, accumulate 'precision', 'recall', and 'f1-score' for each label
    for report in reports:
        acc.append(report['accuracy'])
        roc_auc.append(report['roc_auc'])
        for label in labels:
            label = str(label)  # Ensure label is string
            if label in report:
                total[label]['precision'] += report[label]['precision']
                total[label]['recall'] += report[label]['recall']
                total[label]['f1-score'] += report[label]['f1-score']

    # Calculate average values
    num_reports = len(reports)
    avg_report = {str(label): {} for label in labels}
    for label in labels:
        label = str(label)  # Ensure label is string
        avg_report[label]['precision'] = total[label]['precision'] / num_reports
        avg_report[label]['recall'] = total[label]['recall'] / num_reports
        avg_report[label]['f1-score'] = total[label]['f1-score'] / num_reports

    # Calculate average accuracy
    avg_acc = np.mean(acc)

    # Calculate average ROC-AUC
    avg_roc_auc = np.mean(roc_auc)

    return avg_report, avg_acc, avg_roc_auc

def print_classification_results(y_true, y_pred, y_pred_proba=None):
    '''
    Print classification results

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities, used for calculating ROC-AUC

    Returns:
        report_dict: Classification report dictionary
    '''
    # Generate classification report
    report_text = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    
    print("Classification report:")
    print(report_text)
    
    # If prediction probabilities are provided, calculate ROC-AUC
    if y_pred_proba is not None:
        # Check if binary or multi-class classification
        num_classes = len(np.unique(y_true))
        if num_classes == 2:
            # Binary classification case
            try:
                # Try to get positive class probability
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                print(f"ROC-AUC value: {roc_auc:.4f}")
                report_dict['roc_auc'] = roc_auc
            except IndexError:
                # If y_pred_proba is a 1D array
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                print(f"ROC-AUC value: {roc_auc:.4f}")
                report_dict['roc_auc'] = roc_auc
        else:
            # Multi-class classification case
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
            print(f"Multi-class ROC-AUC value: {roc_auc:.4f}")
            report_dict['roc_auc'] = roc_auc
    
    return report_dict

def summarize_k_fold_results(reports, labels):
    '''
    Summarize K-fold cross-validation results

    Args:
        reports: List containing evaluation results for each fold
        labels: List of labels to calculate average for

    Returns:
        None
    '''
    avg_report, avg_acc, avg_roc_auc = calculate_average_reports(reports, labels)
    
    print("\n======= K-fold cross-validation average results =======")
    print(f"Average accuracy: {avg_acc:.4f}")
    print(f"Average ROC-AUC value: {avg_roc_auc:.4f}")
    print("\nAverage performance metrics per class:")
    
    for label in labels:
        label_str = str(label)
        print(f"Class {label}:")
        print(f"  Precision: {avg_report[label_str]['precision']:.4f}")
        print(f"  Recall: {avg_report[label_str]['recall']:.4f}")
        print(f"  F1-Score: {avg_report[label_str]['f1-score']:.4f}")
    
    return avg_report, avg_acc, avg_roc_auc