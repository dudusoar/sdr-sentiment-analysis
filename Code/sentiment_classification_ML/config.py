# coding: utf-8
'''
filename: config.py
function: 集中管理配置参数
'''

import os
import string
from nltk.corpus import stopwords

# 路径配置
DATA_DIR = 'data'
RESULTS_DIR = 'results'
DATASET_DIR = os.path.join(RESULTS_DIR, 'dataset')
WORD_FREQ_DIR = os.path.join(RESULTS_DIR, 'word_frequency_results')

# 数据文件
COMBINED_COMMENTS_FILE = os.path.join(DATA_DIR, 'combined_comments.xlsx')
SELECTED_COMMENTS_FILE = os.path.join(RESULTS_DIR, 'selected_comments.xlsx')
WORD2VEC_MODEL_PATH = os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin')

# 预处理配置
# 定义需要替换的缩写形式及其对应的全称形式
CONTRACTIONS = {
    # n't
    "didn't": "did not",
    "don't": "do not",
    "doesn't": "does not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    # be动词
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "I'm": "I am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    # 've
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "I've": "I have",
    # 其他省略形式
    "kinda": "kind of",
    "sorta": "sort of",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "hafta": "have to",
    "needa": "need to",
    "outta": "out of",
    "lemme": "let me",
    "gimme": "give me"
}

# 停用词
STOP_WORDS = set(stopwords.words('english')).copy() | set(['\'s', 'n\'t','lol','\'m','\'re','\'d','\'ve'])

# 额外需要去除的无意义词
ADDITIONAL_STOP_WORDS = {'to', 'from', 'if', 'would', 'could', 'now', 'one', 'someone', 'thing', 'many', 'even', 'already', 'much'}

# 添加额外的停用词
STOP_WORDS.update(ADDITIONAL_STOP_WORDS)

# 自定义保留词
RETAIN_WORDS = {'not', 'no', 'but', 'while', 'have', 'into', 'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose'}
# 更新停用词列表，移除保留词
for word in RETAIN_WORDS:
    STOP_WORDS.discard(word)

# 标点符号
# 所有标点符号

PUNCTUATION = set(string.punctuation) | set(['...','``','\'\'','’','..','....','.....','“','”'])
# 需要保留的标点符号
PUNCTUATION_PRESERVED = set([',','!','?','.'])
# 需要移除的标点符号
PUNCTUATION_REMOVED = PUNCTUATION - PUNCTUATION_PRESERVED

# 模型训练配置
RANDOM_SEED = 42
TEST_SIZE = 0.2
K_FOLDS = 10

# 创建必要的目录
def create_directories():
    """创建必要的文件夹"""
    for directory in [DATA_DIR, RESULTS_DIR, DATASET_DIR, WORD_FREQ_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")