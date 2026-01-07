# coding: utf-8
'''
filename: utils.py
function: 通用工具函数
'''

import os
import pandas as pd
import re
from config import CONTRACTIONS

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建目录: {directory}")

def save_excel(df, filepath):
    """保存DataFrame到Excel文件"""
    directory = os.path.dirname(filepath)
    ensure_directory(directory)
    df.to_excel(filepath, index=False)
    print(f"文件已保存: {filepath}")

def load_excel(filepath):
    """加载Excel文件到DataFrame"""
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return None
    return pd.read_excel(filepath)

def word_replace(text):
    """替换文本中的缩写形式为全称形式"""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

def is_digit(word):
    """检查单词是否为数字"""
    return bool(re.match(r'\d+', word))