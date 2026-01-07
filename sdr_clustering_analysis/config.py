"""
SDR Clustering Analysis Configuration File
"""

import os
import string
from nltk.corpus import stopwords

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root directory: sdr_clustering_analysis/
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CLUSTER_PROFILES_DIR = os.path.join(RESULTS_DIR, 'cluster_profiles')

# --- Data Files ---
RAW_COMMENTS_FILE = os.path.join(DATA_DIR, 'combined_comments.xlsx')
PREPROCESSED_COMMENTS_FILE = os.path.join(DATA_DIR, 'preprocessed_sdr_comments.pkl')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'sbert_embeddings_sdr.pkl')

# --- Text Preprocessing Configuration ---
TEXT_COLUMN = 'pure_text'              # Original comment text
CLEANED_TEXT_COLUMN = 'cleaned_text'   # Cleaned text
ID_COLUMN = 'index'                    # Use index as unique identifier
MANUAL_SENTIMENT_COLUMN = 'label1'     # Sentiment label column

# Stop words configuration (consistent with the sentiment classification project)
STOP_WORDS = set(stopwords.words('english')).copy() | set([
    '\'s', 'n\'t', 'lol', '\'m', '\'re', '\'d', '\'ve'
])
ADDITIONAL_STOP_WORDS = {
    'to', 'from', 'if', 'would', 'could', 'now', 'one',
    'someone', 'thing', 'many', 'even', 'already', 'much'
}
STOP_WORDS.update(ADDITIONAL_STOP_WORDS)

# Words explicitly retained (not removed as stop words)
RETAIN_WORDS = {
    'not', 'no', 'but', 'while', 'have', 'into',
    'who', 'what', 'where', 'when', 'why', 'how',
    'which', 'whose'
}
for word in RETAIN_WORDS:
    STOP_WORDS.discard(word)

# Punctuation configuration (consistent with the sentiment classification project)
PUNCTUATION = set(string.punctuation) | {
    '...', '``', '\'\'', '\'', '..', '....', '.....', '"'
}
PUNCTUATION_PRESERVED = {',', '!', '?', '.'}
PUNCTUATION_REMOVED = PUNCTUATION - PUNCTUATION_PRESERVED

# --- Feature Extraction Configuration ---
# Use a multilingual model because the dataset may contain non-English comments
SBERT_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SBERT_DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# --- Clustering Algorithm Configuration ---
# Adjust the clustering range based on the sentiment label distribution (5 major categories)
KMEANS_N_CLUSTERS_RANGE = range(3, 8)  # Test K = 3, 4, 5, 6, 7
KMEANS_SELECTED_K = None              # Set to None for automatic selection of optimal K
KMEANS_RANDOM_STATE = 42

# HDBSCAN parameter tuning
HDBSCAN_MIN_CLUSTER_SIZE = 20  # Considering dataset size (6,976 comments)
HDBSCAN_MIN_SAMPLES = 10

# --- Output Configuration ---
CLUSTER_ASSIGNMENT_FILE = os.path.join(RESULTS_DIR, 'cluster_assignments_sdr.csv')
CLUSTER_EVALUATION_FILE = os.path.join(RESULTS_DIR, 'cluster_evaluation_sdr.txt')
CLUSTER_SENTIMENT_CROSSTAB_FILE = os.path.join(
    RESULTS_DIR, 'cluster_sentiment_crosstab_sdr.csv'
)
CLUSTER_LDA_CROSSTAB_FILE = os.path.join(
    RESULTS_DIR, 'cluster_lda_crosstab_sdr.csv'
)

# --- Miscellaneous ---
RANDOM_SEED = 42

# Data configuration
DATA_CONFIG = {
    'input_file': 'data/combined_comments.xlsx',
    'text_column': 'pure_text',
    'min_text_length': 3,     # Filter out very short comments
    'max_text_length': 500    # Longest comment is ~756 chars; set a reasonable upper bound
}

# Text preprocessing configuration
PREPROCESSING_CONFIG = {
    'min_token_length': 2,
    'max_features': 2000,     # Increase feature count due to rich comment content
    'min_df': 3,              # Lower minimum document frequency to retain rare but important words
    'max_df': 0.9,            # Filter out overly frequent words
    'stop_words': STOP_WORDS,
    'punctuation_removed': PUNCTUATION_REMOVED,
    'punctuation_preserved': PUNCTUATION_PRESERVED
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'n_components': 128,      # Increase reduced dimensionality to preserve more information
    'batch_size': 32          # Batch size for embedding computation
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'n_clusters_range': range(3, 8),  # Consistent with KMEANS_N_CLUSTERS_RANGE
    'random_state': 42,
    'max_iter': 300                   # Increase maximum number of iterations
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
    'min_cluster_size': 50             # Minimum acceptable cluster size
}

# Output settings
OUTPUT_CONFIG = {
    'results_dir': 'results/cluster_profiles',
    'log_dir': 'logs',
    'save_embeddings': True,           # Save embeddings for further analysis
    'save_cluster_samples': True       # Save sample texts from each cluster
}

# Create required directories
def create_project_directories():
    """Create all required output directories for the project."""
    dirs_to_create = [
        RESULTS_DIR,
        os.path.join(RESULTS_DIR, 'visualizations')
    ]
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == '__main__':
    print(f"Project root directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    create_project_directories()
    print("Configuration loaded. Directories checked/created.")
