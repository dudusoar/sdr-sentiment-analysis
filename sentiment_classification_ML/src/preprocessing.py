# coding: utf-8
'''
filename: preprocessing.py
function: Text preprocessing related functions
'''

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from config import STOP_WORDS, PUNCTUATION_REMOVED, PUNCTUATION_PRESERVED
from utils import is_digit, word_replace

# Stemmer
stemmer = PorterStemmer()
# Lemmatizer
lemmatizer = WordNetLemmatizer()

def first_preprocessing(text, filtering=True):
    '''
    First stage of text preprocessing: case conversion, tokenization, removing useless punctuation, removing numbers, removing stop words
    
    Args:
        text: Original text
        filtering: Whether to remove stop words (some custom stop words are retained and not removed)
    
    Returns:
        words: Processed word list
    '''
    if type(text) != str:
        text = ''
    
    text = text.lower()
    #text = word_replace(text)  # Uncomment to use contraction replacement feature

    words = word_tokenize(text)
    words = [word for word in words if not is_digit(word)]
    words = [w for w in words if w not in PUNCTUATION_REMOVED]
    if filtering:
        words = [w for w in words if w not in STOP_WORDS]
    return words

def second_preprocessing(words, remove_punctuation=True, lemmatization=True):
    '''
    Second stage of text preprocessing: removing punctuation, part-of-speech restoration
    
    Args:
        words: Word list processed by first_preprocessing(text)
        remove_punctuation: Whether to remove punctuation (at this stage, punctuation only includes [',', '!', '?', '.'])
        lemmatization: Whether to perform lemmatization
    
    Returns:
        words: Further processed word list
    '''
    if remove_punctuation:
        words = [w for w in words if w not in PUNCTUATION_PRESERVED]
    
    if lemmatization:
        words = [lemmatizer.lemmatize(w) for w in words]
    
    return words

def remove_low_frequency_words(words, low_frequency_words):
    '''
    Remove low-frequency words
    
    Args:
        words: Word list
        low_frequency_words: List of low-frequency words
    
    Returns:
        filtered_words: Word list after removing low-frequency words
    '''
    return [w for w in words if w not in low_frequency_words]