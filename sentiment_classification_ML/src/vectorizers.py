# coding: utf-8
'''
filename: vectorizers.py
function: Text vectorization
'''

import numpy as np

class Word2VecVectorizer:
    def __init__(self, model, bow='avg', shift_to_positive=False):
        '''
        Word2Vec-based text vectorizer
        
        Args:
            model: Pre-trained Word2Vec model
            bow: 'sum' or 'avg', method for aggregating word vectors
            shift_to_positive: Whether to shift word vectors so all elements are positive
        '''
        self.model = model
        self.dim = model.vector_size
        self.bow = bow
        self.min_val = None
        self.shift_to_positive = shift_to_positive

    def fit_transform(self, X):
        '''
        Train and transform text data into vectors
        
        Args:
            X: List of word lists
        
        Returns:
            transformed_X: Vectorized data
        '''
        transformed_X = np.array([self.word_list_to_vec(words) for words in X])
        # Shift word vectors
        if self.shift_to_positive:
            self.min_val = transformed_X.min()
        return self.shift_vectors(transformed_X)

    def transform(self, X):
        '''
        Transform text data into vectors
        
        Args:
            X: List of word lists
        
        Returns:
            transformed_X: Vectorized data
        '''
        transformed_X = np.array([self.word_list_to_vec(words) for words in X])
        # Shift word vectors
        if self.shift_to_positive:
            # Ensure all elements in test set vectors are positive after shifting
            min_val = transformed_X.min()
            if self.min_val > min_val:
                self.min_val = min_val
        return self.shift_vectors(transformed_X)
    
    def word_list_to_vec(self, word_list):
        '''
        Convert a word list into a vector
        
        Args:
            word_list: List of words
        
        Returns:
            word_vec: Vector representation
        '''
        n = len(word_list)
        word_matrix = np.zeros([n, self.dim])
        for i in range(n):
            # Skip words not present in the pre-trained model
            try:
                word_matrix[i, :] = self.model[word_list[i]]
            except KeyError:
                continue
        
        if self.bow == 'sum':
            word_vec = word_matrix.sum(axis=0)
        elif self.bow == 'avg':
            word_vec = word_matrix.mean(axis=0)
        else:
            raise ValueError("bow parameter must be 'sum' or 'avg'")
        
        return word_vec
    
    def shift_vectors(self, vectors):
        '''
        Shift vectors so all elements are positive
        
        Args:
            vectors: Array of vectors
        
        Returns:
            shifted_vectors: Shifted vector array
        '''
        if self.shift_to_positive and self.min_val is not None and self.min_val < 0:
            return vectors - self.min_val
        else:
            return vectors