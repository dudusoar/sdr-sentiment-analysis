# Analysis Pipeline

This document describes the end-to-end analysis pipeline implemented in the YouTube-SC project, from raw YouTube comments to insights about public perceptions of sidewalk delivery robots.

## Overview
The research follows a three-stage pipeline designed for comprehensive analysis of YouTube comments:
1. **Data Collection & Preprocessing**: Raw comment collection and text preparation
2. **Sentiment Classification**: Multi-method sentiment analysis (ML and deep learning)
3. **Topic Modeling & Interpretation**: Theme extraction and qualitative analysis

## Stage 1: Data Collection & Preprocessing

### 1.1 YouTube Comment Collection
- Comments extracted from publicly available YouTube videos about delivery robots
- Using YouTube Data API with proper authentication
- Collection includes comment text, timestamps, and engagement metrics

### 1.2 Text Preprocessing Pipeline
The following preprocessing steps are applied to each comment:

**Case Folding**
- Convert all text to lowercase for consistency

**Tokenization**
- Split text into individual tokens/words
- Handle punctuation and special characters appropriately

**Stopword Filtering**
- Remove common English stopwords using NLTK stopword list
- Preserve negation words (not, never, no, etc.) for sentiment analysis

**Lemmatization**
- Reduce words to their base/dictionary form using WordNet lemmatizer
- Maintains semantic meaning while reducing dimensionality

**Special Handling**
- Preservation of negation words and selected punctuation for sentiment context
- Emoji and emoticon handling (conversion to text representations)

### 1.3 Data Annotation
- Manual annotation of each comment into three sentiment categories:
  - **0**: Negative sentiment
  - **1**: Positive sentiment
  - **2**: Neutral sentiment
- Annotation performed by trained annotators with inter-rater reliability checks

## Stage 2: Sentiment Classification

### 2.1 Feature Engineering
**TF-IDF Features**
- Term Frequency-Inverse Document Frequency vectors
- Configurable n-gram ranges (unigrams, bigrams, trigrams)
- Vocabulary size optimization for performance

**Word Embeddings**
- Word2Vec embeddings trained on Google News corpus
- Fallback mechanisms for missing embeddings
- Alternative: FastText or GloVe embeddings

**Deep Learning Features**
- BERT tokenizer and embeddings for transformer-based models
- Sequence padding and attention masking

### 2.2 Machine Learning Models
Multiple traditional ML algorithms are implemented with configurable strategies:

**Models**
- **Support Vector Machine (SVM)**: RBF kernel with parameter tuning
- **Naive Bayes**: Multinomial variant for text classification
- **Random Forest**: Ensemble method with feature importance
- **Decision Tree**: Baseline model for interpretability

**Classification Strategies**
- **One-vs-One (OVO)**: Train binary classifiers for each pair of classes
- **One-vs-Rest (OVR)**: Train binary classifiers for each class vs all others

**Performance Metrics**
- Accuracy, Precision, Recall, F1-score (weighted)
- ROC-AUC for binary classification tasks
- Confusion matrix analysis

### 2.3 Deep Learning Models
**BERT-based Architecture**
- Base model: `bert-base-uncased` (English BERT)
- Additional layers: Bidirectional LSTM and GRU for sequence modeling
- Dropout regularization and linear classification head

**Training Configuration**
- Optimizer: AdamW with weight decay
- Learning rate scheduling with warmup
- Early stopping based on validation loss
- Batch size optimization (32, 64, 128)

**Fine-tuning**
- Transfer learning from pre-trained BERT weights
- Gradual unfreezing of transformer layers
- Custom loss functions for class imbalance

## Stage 3: Topic Modeling & Interpretation

### 3.1 Latent Dirichlet Allocation (LDA)
**Model Configuration**
- Number of topics: 10 (optimized via coherence scores)
- Dirichlet priors: α (document-topic), β (topic-word)
- Gibbs sampling for parameter estimation

**Preprocessing for Topic Modeling**
- Additional stopword removal (domain-specific terms)
- Phrase detection and bigram formation
- Frequency filtering (remove very rare and very common terms)

### 3.2 Topic Evaluation
**Coherence Measures**
- C_v coherence for topic interpretability
- UMass coherence for model comparison
- Perplexity evaluation on held-out data

**Human Evaluation**
- Manual labeling of topics based on top keywords
- Topic consistency assessment by domain experts
- Cross-validation with different random seeds

### 3.3 Interpretation Framework
**Keyword Analysis**
- Top-N words per topic with probabilities
- Word salience and distinctiveness measures
- Topic-word distribution visualization

**Topic Labeling**
- Manual assignment of descriptive labels to topics
- Grouping related topics into thematic categories
- Mapping topics to research questions about SDR perceptions

## Integrated Analysis Modules

### Clustering Analysis (`sdr_clustering_analysis/`)
- **Algorithm**: K-means clustering with elbow method for K selection
- **Features**: TF-IDF vectors or word embeddings
- **Application**: Discover natural comment groups beyond sentiment
- **Visualization**: 2D/3D cluster plots using PCA/t-SNE

### Text Statistics (`yearly_word_frequency/`)
- **Word Frequency Analysis**: Most common terms by year
- **Trend Analysis**: Changing vocabulary over time
- **Statistical Metrics**: Readability scores, sentiment distributions
- **Visualization**: Frequency plots, heatmaps, trend lines

## Data Flow Through Modules

```
Raw Comments (data/combined_comments.xlsx)
        ↓
    [Preprocessing & Annotation]
        ↓
┌─────────────────────────────┐
│  sentiment_classification_ML │ → Sentiment labels & performance metrics
│  sentiment_classification_Bert│ → Deep learning sentiment predictions
│  sdr_clustering_analysis    │ → Comment clusters & patterns
│  topic_modeling             │ → Discussion topics & themes
│  yearly_word_frequency      │ → Statistical trends & metrics
└─────────────────────────────┘
        ↓
Results saved to respective module's `results/` directories
        ↓
Integrated insights about public perceptions of SDRs
```

## Pipeline Configuration

Each module has its own `config.py` file for managing:
- Data paths and file locations
- Model hyperparameters
- Feature extraction settings
- Output directory structures

## Reproducibility
- Random seeds fixed for reproducible results
- Configuration files capture all experimental settings
- Detailed logging of preprocessing steps and model parameters
- Version control of data and code for exact replication

## Performance Optimization
- **Memory**: Sparse matrices for TF-IDF features, batch processing for large datasets
- **Speed**: Parallel processing, feature caching, GPU acceleration for deep learning
- **Accuracy**: Ensemble methods, hyperparameter tuning, cross-validation

This pipeline provides a comprehensive framework for analyzing public perceptions of sidewalk delivery robots through multiple complementary approaches.