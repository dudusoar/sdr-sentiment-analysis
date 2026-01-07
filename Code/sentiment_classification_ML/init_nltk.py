#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
初始化NLTK数据下载脚本
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
    """下载必要的NLTK数据"""
    datasets = ['stopwords', 'punkt_tab', 'punkt', 'wordnet']
    
    for dataset in datasets:
        print(f"正在下载NLTK {dataset}数据...")
        try:
            nltk.download(dataset, quiet=True)
            print(f"NLTK {dataset}数据下载完成!")
        except Exception as e:
            print(f"下载{dataset}失败: {e}")
            # 继续下载其他数据集
            continue
    
    print("所有NLTK数据下载完成!")
    return True

if __name__ == "__main__":
    download_nltk_data() 