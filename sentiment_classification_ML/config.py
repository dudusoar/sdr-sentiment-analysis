# coding: utf-8
'''
filename: config.py
function: Centralized configuration parameter management
'''

import os
import string
from nltk.corpus import stopwords

# Path Configuration
DATA_DIR = 'data'
RESULTS_DIR = 'results'
DATASET_DIR = os.path.join(RESULTS_DIR, 'dataset')
WORD_FREQ_DIR = os.path.join(RESULTS_DIR, 'word_frequency_results')

# Data Files
COMBINED_COMMENTS_FILE = os.path.join(DATA_DIR, 'combined_comments.xlsx')
SELECTED_COMMENTS_FILE = os.path.join(RESULTS_DIR, 'selected_comments.xlsx')
WORD2VEC_MODEL_PATH = os.path.join(DATA_DIR, 'GoogleNews-vectors-negative300.bin')

# Preprocessing Configuration
# Define contractions to be replaced and their corresponding full forms
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
    # 're (be verb)
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
    # Other informal forms
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

# Stopwords
STOP_WORDS = set(stopwords.words('english')).copy() | set(['\'s', 'n\'t', 'lol', '\'m', '\'re', '\'d', '\'ve'])

# Additional meaningless words to remove
ADDITIONAL_STOP_WORDS = {'to', 'from', 'if', 'would', 'could', 'now', 'one', 'someone', 'thing', 'many', 'even', 'already', 'much'}

# Add additional stop words
STOP_WORDS.update(ADDITIONAL_STOP_WORDS)

# Custom words to retain
RETAIN_WORDS = {'not', 'no', 'but', 'while', 'have', 'into', 'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose'}
# Update the stop words list by removing the words to retain
for word in RETAIN_WORDS:
    STOP_WORDS.discard(word)

# Punctuation
# All punctuation marks
PUNCTUATION = set(string.punctuation) | set(['...', '``', '\'\'', '’', '..', '....', '.....', '“', '”'])
# Punctuation marks to preserve
PUNCTUATION_PRESERVED = set([',', '!', '?', '.'])
# Punctuation marks to remove
PUNCTUATION_REMOVED = PUNCTUATION - PUNCTUATION_PRESERVED

# Model Training Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
K_FOLDS = 10

# Create necessary directories
def create_directories():
    """Create necessary folders"""
    for directory in [DATA_DIR, RESULTS_DIR, DATASET_DIR, WORD_FREQ_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")