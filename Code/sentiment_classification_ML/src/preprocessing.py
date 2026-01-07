# coding: utf-8
'''
filename: preprocessing.py
function: 文本预处理相关函数
'''

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from config import STOP_WORDS, PUNCTUATION_REMOVED, PUNCTUATION_PRESERVED
from utils import is_digit, word_replace

# 词干提取器
stemmer = PorterStemmer()
# 词形还原器
lemmatizer = WordNetLemmatizer()

def first_preprocessing(text, filtering=True):
    '''
    文本第一阶段预处理：大小写、分词、去除无用的标点符号、去除数字、去除停用词
    
    Args:
        text: 原始文本
        filtering: 是否去除停用词（保留一部分自定义的停用词不被去除）
    
    Returns:
        words: 处理后的单词列表
    '''
    if type(text) != str:
        text = ''
    
    text = text.lower()
    #text = word_replace(text)  # 取消注释以使用缩写替换功能

    words = word_tokenize(text)
    words = [word for word in words if not is_digit(word)]
    words = [w for w in words if w not in PUNCTUATION_REMOVED]
    if filtering:
        words = [w for w in words if w not in STOP_WORDS]
    return words

def second_preprocessing(words, remove_punctuation=True, lemmatization=True):
    '''
    文本第二阶段预处理：去除标点符号、词性恢复
    
    Args:
        words: 经过first_preprocessing(text)处理后的单词列表
        remove_punctuation: 是否去掉标点符号，这时的标点只包含[',','!','?','.']
        lemmatization: 是否进行词形还原
    
    Returns:
        words: 进一步处理后的单词列表
    '''
    if remove_punctuation:
        words = [w for w in words if w not in PUNCTUATION_PRESERVED]
    
    if lemmatization:
        words = [lemmatizer.lemmatize(w) for w in words]
    
    return words

def remove_low_frequency_words(words, low_frequency_words):
    '''
    去除低频词
    
    Args:
        words: 单词列表
        low_frequency_words: 低频词列表
    
    Returns:
        filtered_words: 去除低频词后的单词列表
    '''
    return [w for w in words if w not in low_frequency_words]