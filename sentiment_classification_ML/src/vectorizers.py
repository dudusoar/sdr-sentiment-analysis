# coding: utf-8
'''
filename: vectorizers.py
function: 文本向量化
'''

import numpy as np

class Word2VecVectorizer:
    def __init__(self, model, bow='avg', shift_to_positive=False):
        '''
        基于Word2Vec的文本向量化器
        
        Args:
            model: 预训练好的Word2Vec模型
            bow: 'sum'或'avg'，词向量的聚合方式
            shift_to_positive: 是否平移词向量使所有元素为正值
        '''
        self.model = model
        self.dim = model.vector_size
        self.bow = bow
        self.min_val = None
        self.shift_to_positive = shift_to_positive

    def fit_transform(self, X):
        '''
        训练并转换文本数据到向量
        
        Args:
            X: 单词列表的列表
        
        Returns:
            transformed_X: 向量化后的数据
        '''
        transformed_X = np.array([self.word_list_to_vec(words) for words in X])
        # 平移词向量
        if self.shift_to_positive:
            self.min_val = transformed_X.min()
        return self.shift_vectors(transformed_X)

    def transform(self, X):
        '''
        将文本数据转换为向量
        
        Args:
            X: 单词列表的列表
        
        Returns:
            transformed_X: 向量化后的数据
        '''
        transformed_X = np.array([self.word_list_to_vec(words) for words in X])
        # 平移词向量
        if self.shift_to_positive:
            # 为了确保test集的词向量平移后各个元素都能大于0
            min_val = transformed_X.min()
            if self.min_val > min_val:
                self.min_val = min_val
        return self.shift_vectors(transformed_X)
    
    def word_list_to_vec(self, word_list):
        '''
        将单词列表转换为一个向量
        
        Args:
            word_list: 单词列表
        
        Returns:
            word_vec: 向量表示
        '''
        n = len(word_list)
        word_matrix = np.zeros([n, self.dim])
        for i in range(n):
            # 有些词在预训练的模型中不存在，跳过
            try:
                word_matrix[i, :] = self.model[word_list[i]]
            except KeyError:
                continue
        
        if self.bow == 'sum':
            word_vec = word_matrix.sum(axis=0)
        elif self.bow == 'avg':
            word_vec = word_matrix.mean(axis=0)
        else:
            raise ValueError("bow参数必须是'sum'或'avg'")
        
        return word_vec
    
    def shift_vectors(self, vectors):
        '''
        平移向量使所有元素为正值
        
        Args:
            vectors: 向量数组
        
        Returns:
            shifted_vectors: 平移后的向量数组
        '''
        if self.shift_to_positive and self.min_val is not None and self.min_val < 0:
            return vectors - self.min_val
        else:
            return vectors