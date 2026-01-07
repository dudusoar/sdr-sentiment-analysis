# coding: utf-8
'''
filename: data_manager.py
function: 数据读取、预处理、分割
'''

import pandas as pd
import os
from src.preprocessing import first_preprocessing, second_preprocessing, remove_low_frequency_words
from config import DATASET_DIR, SELECTED_COMMENTS_FILE
from utils import save_excel, load_excel, ensure_directory

def load_data(file_name=SELECTED_COMMENTS_FILE, remove_punctuation=True, remove_low_frequency=True, low_frequency_words=None):
    '''
    加载数据并进行预处理
    
    Args:
        file_name: 数据文件路径
        remove_punctuation: 是否去除标点符号
        remove_low_frequency: 是否去除低频词
        low_frequency_words: 低频词列表
    
    Returns:
        comments: 处理后的DataFrame
    '''
    comments = load_excel(file_name)
    if comments is None:
        return None
    
    # 小写、分词、去除无用的标点符号、去除数字、去除停用词
    comments['f_text'] = comments['pure_text'].apply(first_preprocessing)
    # 保留停用词的版本
    comments['k_text'] = comments['pure_text'].apply(first_preprocessing, filtering=False)

    # 根据参数决定是否去除标点符号
    comments['f_word_list'] = comments['f_text'].apply(
        second_preprocessing, 
        remove_punctuation=remove_punctuation, 
        lemmatization=True
    )
    comments['k_word_list'] = comments['k_text'].apply(
        second_preprocessing, 
        remove_punctuation=remove_punctuation, 
        lemmatization=True
    )
    
    # 根据参数决定是否去除低频词
    if remove_low_frequency and low_frequency_words is not None:
        comments['f_word_list'] = comments['f_word_list'].apply(
            remove_low_frequency_words, 
            low_frequency_words=low_frequency_words
        )
        comments['k_word_list'] = comments['k_word_list'].apply(
            remove_low_frequency_words, 
            low_frequency_words=low_frequency_words
        )
    
    # 过滤掉空列表
    mask = comments['f_word_list'].apply(lambda x: x != [])
    comments = comments[mask]
    comments = comments.reset_index(drop=True)  # 重置索引

    return comments

def filter_selected_comments(input_file, output_file=SELECTED_COMMENTS_FILE):
    '''
    将标签为0，1，2的数据筛选出来
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    
    Returns:
        selected_comments: 筛选后的DataFrame
    '''
    comments = load_excel(input_file)
    if comments is None:
        return None
    
    index = comments['label1'].isin([0, 1, 2])
    selected_comments = comments[index]
    save_excel(selected_comments, output_file)
    
    return selected_comments

def create_ovo_datasets(data):
    '''
    生成一对一(OVO)策略的数据集
    
    Args:
        data: 包含标签的DataFrame
    
    Returns:
        None
    '''
    ensure_directory(DATASET_DIR)
    
    index_01 = data['label1'].isin([0, 1])
    index_02 = data['label1'].isin([0, 2])
    index_12 = data['label1'].isin([1, 2])

    data_01 = data[index_01].reset_index(drop=True)
    data_02 = data[index_02].reset_index(drop=True)
    data_12 = data[index_12].reset_index(drop=True)

    # 保存文件
    save_excel(data_01, os.path.join(DATASET_DIR, 'data_01.xlsx'))
    save_excel(data_02, os.path.join(DATASET_DIR, 'data_02.xlsx'))
    save_excel(data_12, os.path.join(DATASET_DIR, 'data_12.xlsx'))

def create_ovr_datasets(data):
    '''
    生成一对多(OVR)策略的数据集
    
    Args:
        data: 包含标签的DataFrame
    
    Returns:
        None
    '''
    ensure_directory(DATASET_DIR)
    
    # 保留原有的标签，其余标签都标注为9
    data_0 = data.copy()
    data_0['label1'] = data_0['label1'].apply(lambda x: 0 if x == 0 else 9)

    data_1 = data.copy()
    data_1['label1'] = data_1['label1'].apply(lambda x: 1 if x == 1 else 9)

    data_2 = data.copy()
    data_2['label1'] = data_2['label1'].apply(lambda x: 2 if x == 2 else 9)

    # 保存文件
    save_excel(data_0, os.path.join(DATASET_DIR, 'data_0.xlsx'))
    save_excel(data_1, os.path.join(DATASET_DIR, 'data_1.xlsx'))
    save_excel(data_2, os.path.join(DATASET_DIR, 'data_2.xlsx'))

def create_datasets():
    '''
    主函数：创建所有数据集
    '''
    # 筛选数据
    selected_comments = filter_selected_comments('combined_comments.xlsx')
    if selected_comments is None:
        print("无法加载原始数据文件")
        return
    
    # 创建OVO和OVR数据集
    create_ovo_datasets(selected_comments)
    create_ovr_datasets(selected_comments)
    print("所有数据集创建完成")

def load_training_datasets(type='ovo', remove_punctuation=False, remove_low_frequency=True, low_frequency_words=None):
    '''
    加载训练用数据集
    
    Args:
        type: 'ovo'或'ovr'
        remove_punctuation: 是否去除标点符号
        remove_low_frequency: 是否去除低频词
        low_frequency_words: 低频词列表
    
    Returns:
        datasets: 数据集列表
        datasets_names: 数据集名称列表
    '''
    if type == 'ovo':
        # OVO数据集
        data_01_r = load_data(os.path.join(DATASET_DIR, 'data_01.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_01_k = load_data(os.path.join(DATASET_DIR, 'data_01.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        data_02_r = load_data(os.path.join(DATASET_DIR, 'data_02.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_02_k = load_data(os.path.join(DATASET_DIR, 'data_02.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        data_12_r = load_data(os.path.join(DATASET_DIR, 'data_12.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=remove_low_frequency,
                             low_frequency_words=low_frequency_words)
        data_12_k = load_data(os.path.join(DATASET_DIR, 'data_12.xlsx'), 
                             remove_punctuation=remove_punctuation, 
                             remove_low_frequency=False,
                             low_frequency_words=low_frequency_words)
        
        datasets = [data_01_r, data_01_k, data_02_r, data_02_k, data_12_r, data_12_k]
        datasets_names = ['data_01_r', 'data_01_k', 'data_02_r', 'data_02_k', 'data_12_r', 'data_12_k']
    
    elif type == 'ovr':
        # OVR数据集
        data_0_r = load_data(os.path.join(DATASET_DIR, 'data_0.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_0_k = load_data(os.path.join(DATASET_DIR, 'data_0.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        data_1_r = load_data(os.path.join(DATASET_DIR, 'data_1.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_1_k = load_data(os.path.join(DATASET_DIR, 'data_1.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        data_2_r = load_data(os.path.join(DATASET_DIR, 'data_2.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=remove_low_frequency,
                            low_frequency_words=low_frequency_words)
        data_2_k = load_data(os.path.join(DATASET_DIR, 'data_2.xlsx'), 
                            remove_punctuation=remove_punctuation, 
                            remove_low_frequency=False,
                            low_frequency_words=low_frequency_words)
        
        datasets = [data_0_r, data_0_k, data_1_r, data_1_k, data_2_r, data_2_k]
        datasets_names = ['data_0_r', 'data_0_k', 'data_1_r', 'data_1_k', 'data_2_r', 'data_2_k']
    
    else:
        # 多分类数据集
        data_r = load_data(SELECTED_COMMENTS_FILE, 
                          remove_punctuation=remove_punctuation, 
                          remove_low_frequency=remove_low_frequency,
                          low_frequency_words=low_frequency_words)
        data_k = load_data(SELECTED_COMMENTS_FILE, 
                          remove_punctuation=remove_punctuation, 
                          remove_low_frequency=False,
                          low_frequency_words=low_frequency_words)
        
        datasets = [data_r, data_k]
        datasets_names = ['data_r', 'data_k']
    
    return datasets, datasets_names

if __name__ == '__main__':
    create_datasets()