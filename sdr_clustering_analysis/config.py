"""
SDR Clustering Analysis Configuration File
"""

import os
import string
from nltk.corpus import stopwords

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root directory sdr_clustering_analysis/
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CLUSTER_PROFILES_DIR = os.path.join(RESULTS_DIR, 'cluster_profiles')

# --- Data Files ---
RAW_COMMENTS_FILE = os.path.join(DATA_DIR, 'combined_comments.xlsx')
PREPROCESSED_COMMENTS_FILE = os.path.join(DATA_DIR, 'preprocessed_sdr_comments.pkl')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'sbert_embeddings_sdr.pkl')

# --- Text Preprocessing Configuration ---
TEXT_COLUMN = 'pure_text'  # Original comment text
CLEANED_TEXT_COLUMN = 'cleaned_text'  # Cleaned text
ID_COLUMN = 'index'  # Use index as unique identifier
MANUAL_SENTIMENT_COLUMN = 'label1'  # Sentiment label column

# Stop words configuration (consistent with sentiment classification project)
STOP_WORDS = set(stopwords.words('english')).copy() | set(['\'s', 'n\'t', 'lol', '\'m', '\'re', '\'d', '\'ve'])
ADDITIONAL_STOP_WORDS = {'to', 'from', 'if', 'would', 'could', 'now', 'one', 'someone', 'thing', 'many', 'even', 'already', 'much'}
STOP_WORDS.update(ADDITIONAL_STOP_WORDS)
RETAIN_WORDS = {'not', 'no', 'but', 'while', 'have', 'into', 'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose'}
for word in RETAIN_WORDS:
    STOP_WORDS.discard(word)

# Punctuation configuration (consistent with sentiment classification project)
PUNCTUATION = set(string.punctuation) | set(['...', '``', '\'\'', '\'', '..', '....', '.....', '"', '"'])
PUNCTUATION_PRESERVED = set([',', '!', '?', '.'])
PUNCTUATION_REMOVED = PUNCTUATION - PUNCTUATION_PRESERVED

# --- Feature Extraction Configuration ---
# 使用多语言模型，因为数据可能包含非英语评论
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SBERT_DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# --- 聚类算法配置 ---
# 根据情感标签分布（5个主要类别）调整聚类范围
KMEANS_N_CLUSTERS_RANGE = range(3, 8)  # 测试K=3,4,5,6,7
KMEANS_SELECTED_K = None  # 设置为None以自动选择最佳K值
KMEANS_RANDOM_STATE = 42

# HDBSCAN参数调整
HDBSCAN_MIN_CLUSTER_SIZE = 20  # 考虑到数据规模（6976条评论）
HDBSCAN_MIN_SAMPLES = 10

# --- 结果输出配置 ---
CLUSTER_ASSIGNMENT_FILE = os.path.join(RESULTS_DIR, 'cluster_assignments_sdr.csv')
CLUSTER_EVALUATION_FILE = os.path.join(RESULTS_DIR, 'cluster_evaluation_sdr.txt')
CLUSTER_SENTIMENT_CROSSTAB_FILE = os.path.join(RESULTS_DIR, 'cluster_sentiment_crosstab_sdr.csv')
CLUSTER_LDA_CROSSTAB_FILE = os.path.join(RESULTS_DIR, 'cluster_lda_crosstab_sdr.csv')

# --- 其他 ---
RANDOM_SEED = 42

# 数据配置
DATA_CONFIG = {
    'input_file': 'data/combined_comments.xlsx',
    'text_column': 'pure_text',
    'min_text_length': 3,  # 过滤掉过短的评论
    'max_text_length': 500  # 考虑到最长评论756字符，设置合理的上限
}

# 文本预处理配置
PREPROCESSING_CONFIG = {
    'min_token_length': 2,
    'max_features': 2000,  # 增加特征数量，因为评论内容较丰富
    'min_df': 3,  # 降低最小文档频率，因为有些重要词可能较少出现
    'max_df': 0.9,  # 调整最大文档频率，过滤掉过于常见的词
    'stop_words': STOP_WORDS,
    'punctuation_removed': PUNCTUATION_REMOVED,
    'punctuation_preserved': PUNCTUATION_PRESERVED
}

# 特征提取配置
FEATURE_CONFIG = {
    'n_components': 128,  # 增加降维后的维度，保留更多信息
    'batch_size': 32  # 添加批处理大小配置
}

# 聚类配置
CLUSTERING_CONFIG = {
    'n_clusters_range': range(3, 8),  # 与KMEANS_N_CLUSTERS_RANGE保持一致
    'random_state': 42,
    'max_iter': 300  # 增加最大迭代次数
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'min_cluster_size': 50  # 设置最小聚类大小
}

# 输出配置
OUTPUT_CONFIG = {
    'results_dir': 'results/cluster_profiles',
    'log_dir': 'logs',
    'save_embeddings': True,  # 保存嵌入向量以便后续分析
    'save_cluster_samples': True  # 保存每个聚类的样本
}

# 创建必要的目录
def create_project_directories():
    """创建项目所需的所有结果输出目录"""
    dirs_to_create = [
        RESULTS_DIR,
        os.path.join(RESULTS_DIR, 'visualizations')
    ]
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")

if __name__ == '__main__':
    print(f"项目根目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"结果目录: {RESULTS_DIR}")
    create_project_directories()
    print("配置加载完毕，目录已检查/创建。")