# YouTube Delivery Robot Sentiment Analysis System

## Project Overview

This project is a comprehensive sentiment analysis system specifically designed for analyzing sentiment tendencies in YouTube delivery robot related comments. The system supports three-class sentiment analysis (negative, positive, neutral) and achieves high-precision sentiment recognition through binary classification strategies (OVO and OVR).

### Core Features

- 🤖 **Specialized for delivery robot domain** sentiment analysis
- 📊 **Three-class support**: negative (0), positive (1), neutral (2)
- 🔄 **Dual classification strategies**: OVO (one-vs-one) and OVR (one-vs-rest)
- 🧠 **Multiple ML models**: Naive Bayes, SVM, Random Forest, Decision Tree
- 🔤 **Rich feature extraction**: TF-IDF, Word2Vec (with intelligent fallback mechanism)
- ⚖️ **Data balance handling**: SMOTE oversampling support
- 📈 **Complete performance evaluation**: ROC-AUC, accuracy, F1 score
- 🏗️ **Modular architecture**: Easy maintenance and extension

## Project Structure

```
sentiment_classification_ML/
├── main.py                 # Main program entry point
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── init_nltk.py          # NLTK initialization
├── test_binary_framework.py  # Test scripts
├── src/                   # Core source code
│   ├── binary_classification_framework.py  # Binary classification framework
│   ├── word2vec_downloader.py  # Word2Vec model management
│   ├── data_manager.py    # Data management
│   ├── preprocessing.py   # Data preprocessing
│   ├── vectorizers.py     # Feature vectorization
│   ├── model_training.py  # Model training
│   ├── evaluation.py      # Performance evaluation
│   └── visualization.py   # Result visualization
├── data/                  # Input data files (user-provided)
│   ├── combined_comments.xlsx  # Raw comment data (required)
│   └── GoogleNews-vectors-negative300.bin  # Word2Vec model (optional, auto-downloaded)
├── results/               # Output results
│   ├── selected_comments.xlsx  # Filtered comment data
│   ├── dataset/          # Processed datasets
│   └── word_frequency_results/  # Word frequency analysis results
└── __pycache__/          # Python cache
```

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies from project root using uv
uv pip install -r ../requirements.txt

# Initialize NLTK data (first-time setup)
python init_nltk.py
```

### 2. Input File Preparation

Place the following files in the `data/` directory:

1. **Required**: `combined_comments.xlsx` - Raw comment data with columns:
   - `pure_text`: Comment text
   - `label1`: Sentiment labels (0=negative, 1=positive, 2=neutral)

2. **Optional**: `GoogleNews-vectors-negative300.bin` - Word2Vec model
   - If not present, the system will auto-download when needed
   - File will be automatically downloaded via `--setup-word2vec`

### 3. System Initialization

```bash
# Prepare datasets (creates OVO/OVR datasets)
python main.py --prepare

# Set up Word2Vec model (auto-downloads 1.65GB model)
python main.py --setup-word2vec

# Analyze word frequency
python main.py --analyze
```

### 4. Run Tests

```bash
# Comprehensive binary framework test
python main.py --binary-test comprehensive

# Specific configuration test
python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo

# Run validation script
python test_binary_framework.py
```

## Input File Requirements

### Data File: `combined_comments.xlsx`
- **Location**: `sentiment_classification_ML/data/`
- **Format**: Excel file (.xlsx)
- **Required columns**:
  - `pure_text`: Raw comment text
  - `label1`: Sentiment label (0, 1, or 2)
- **Size**: Approximately 5,000 labeled comments
- **Source**: YouTube delivery robot related comments

### Word2Vec Model: `GoogleNews-vectors-negative300.bin`
- **Status**: Optional, auto-downloaded when needed
- **Size**: ~1.65 GB (full model)
- **Download**: Use `python main.py --setup-word2vec`
- **Fallback**: If download fails, system uses TF-IDF features

## Core Functional Modules

### 1. Binary Classification Framework (`binary_classification_framework.py`)
Complete binary testing framework with:
- **OVO strategy**: One-vs-one classification (data_01, data_02, data_12)
- **OVR strategy**: One-vs-rest classification (data_0, data_1, data_2)
- **Model support**: MultinomialNB, SVM, RandomForest, DecisionTree
- **Feature support**: TF-IDF (1-3 gram), Word2Vec
- **Evaluation metrics**: ROC-AUC, accuracy, F1 score

### 2. Word2Vec Management (`word2vec_downloader.py`)
Automated Word2Vec model management:
- **Auto-download**: Downloads GoogleNews vectors from multiple sources
- **Backup strategies**: Multiple download sources and fallback options
- **Model verification**: Automatic integrity checks
- **Intelligent fallback**: Creates virtual model if full model unavailable
- **Simplified loading**: Easy model loading interface

### 3. Data Management (`data_manager.py`)
Complete data processing pipeline:
- **Data filtering**: Automatically filters comments with labels 0, 1, 2
- **Dataset creation**: Creates OVO and OVR datasets automatically
- **Preprocessing integration**: Low-frequency word filtering, punctuation handling, lemmatization
- **Flexible configuration**: Supports multiple preprocessing options

### 4. Model Training (`model_training.py`)
Unified model training interface:
- **K-fold cross-validation**: Configurable cross-validation
- **Single training**: Quick validation and testing
- **Oversampling support**: SMOTE algorithm integration
- **Performance evaluation**: Automatic calculation of multiple metrics

## Model Performance Summary

Based on comprehensive testing of 4 machine learning models across 6 binary classification tasks, here are the detailed performance metrics:

### Optimal Performance Configuration

```python
# Best configuration: SVM + TF-IDF + OVO strategy
model = SVC(probability=True, kernel='rbf')
vectorizer = TfidfVectorizer(ngram_range=(1,1))
strategy = 'ovo'  # One-vs-one strategy
oversampling = False
remove_low_frequency = True

# Expected performance: ROC-AUC = 0.8946, accuracy = 81%
```

### Detailed Model Performance Analysis

#### 1. Support Vector Machine (SVM) - Best Overall Performer
- **Optimal config**: TF-IDF(1,1) + OVO + No oversampling
- **Top ROC-AUC**: 0.8946 (Negative vs Positive, data_01)
- **Accuracy**: 81% (Negative vs Positive)
- **Key insights**:
  - RBF kernel provides best generalization
  - Word2Vec achieves highest ROC-AUC (0.8977) but requires full model
  - Consistent performance across all binary classification tasks

#### 2. Naive Bayes (MultinomialNB) - Fast and Efficient
- **Optimal config**: TF-IDF(1,1) + OVO + No oversampling
- **Top ROC-AUC**: 0.8876 (Negative vs Positive, data_01)
- **Accuracy**: 82% (Negative vs Positive)
- **Key insights**:
  - Fastest training and inference
  - Excellent precision for positive class (91%)
  - High recall for negative class (97%)
  - Minimal impact from low-frequency word removal

#### 3. Random Forest - Robust and Stable
- **Optimal config**: Word2Vec + OVO + SMOTE oversampling
- **Top ROC-AUC**: 0.8802 (Negative vs Positive, data_01)
- **Accuracy**: 83% (Negative vs Positive)
- **Key insights**:
  - Best performance with Word2Vec features
  - Benefits significantly from SMOTE oversampling
  - Good stability across different configurations
  - n_estimators=100 optimal for this task

#### 4. Decision Tree - Baseline Performance
- **Optimal config**: TF-IDF(1,1) + OVO + No oversampling
- **Top ROC-AUC**: 0.8197 (Negative vs Positive, data_01)
- **Accuracy**: 76% (Negative vs Positive)
- **Key insights**:
  - Simpler model with reasonable performance
  - Performs better with low-frequency words retained
  - Useful for interpretability and debugging

### Performance Comparison Table

| Model | Feature | Strategy | ROC-AUC | Accuracy | Notes |
|-------|---------|----------|---------|----------|-------|
| SVM | TF-IDF(1,1) | OVO | 0.8946 | 81% | **Optimal configuration** ⭐ |
| SVM | Word2Vec | OVO | 0.8977 | 84% | Requires full 1.6GB model |
| Naive Bayes | TF-IDF(1,1) | OVO | 0.8876 | 82% | Fastest inference |
| Random Forest | TF-IDF(1,2) | OVR | 0.8246 | 85% | Best for positive class identification |
| Random Forest | Word2Vec | OVO | 0.8802 | 83% | Best with SMOTE oversampling |
| Decision Tree | TF-IDF(1,1) | OVO | 0.8197 | 76% | Good baseline for comparison |

### Feature Engineering Analysis

#### TF-IDF Features
- **Optimal N-gram range**: (1,1) - unigrams perform best
- **Low-frequency word handling**: Removing low-frequency words generally improves performance
- **Vocabulary size**: Limited to 10,000 features for efficiency
- **Best for**: SVM and Naive Bayes models

#### Word2Vec Features
- **Model requirement**: Full GoogleNews-vectors-negative300.bin (1.6GB)
- **Vector aggregation**: Average word vectors (bow='avg')
- **Performance note**: Virtual model (42 vocabulary) yields ROC-AUC ~0.54 only
- **Best for**: Random Forest with oversampling

### Classification Strategy Comparison

#### OVO (One-vs-One) Strategy
- **Best for**: Balanced binary classification
- **Top performance**: Negative vs Positive (ROC-AUC: 0.89+)
- **Datasets**: data_01, data_02, data_12
- **Advantages**: Better handling of class imbalances in binary pairs

#### OVR (One-vs-Rest) Strategy
- **Best for**: Computational efficiency and large datasets
- **Top performance**: Positive vs Others (ROC-AUC: 0.82+)
- **Datasets**: data_0, data_1, data_2
- **Advantages**: Simpler implementation, faster for multi-class extension

### Classification Difficulty Analysis

#### Difficulty Ranking (Easiest to Hardest)
1. **Negative vs Positive** (data_01): ROC-AUC = 0.89+
   - Clear sentiment distinction
   - Negative class: High recall (96-97%)
   - Positive class: High precision (88-91%)

2. **Positive vs Neutral** (data_12): ROC-AUC = 0.83+
   - Moderate difficulty
   - Neutral class often misclassified as positive
   - Positive class precision: 78-86%

3. **Negative vs Neutral** (data_02): ROC-AUC = 0.80+
   - Most challenging distinction
   - Neutral comments often contain negative sentiment indicators
   - Negative class precision: 68-75%

### Data Balance and Oversampling Impact

#### SMOTE Oversampling Effectiveness
- **Most beneficial for**: Random Forest with Word2Vec features
- **Impact on SVM**: Minimal to negative effect
- **Impact on Naive Bayes**: Limited improvement
- **Recommendation**: Use only with Random Forest models

#### Class Imbalance Analysis
- **Original distribution**: Negative (40%), Neutral (35%), Positive (25%)
- **OVO datasets**: Naturally balanced (~50/50 split)
- **OVR datasets**: Imbalanced (target class vs all others)
- **Best practice**: Use OVO strategy for better balance handling

### Top 5 Configurations by ROC-AUC
1. **SVM + Word2Vec**: 0.8977 (data_01_r) - **Best theoretical performance**
2. **SVM + TF-IDF(1,2)**: 0.8893 (data_01_r) - Best TF-IDF variant
3. **SVM + TF-IDF(1,1)**: 0.8891 (data_01_r) - Recommended production config
4. **Naive Bayes + TF-IDF(1,1)**: 0.8876 (data_01_r) - Best for speed
5. **Random Forest + Word2Vec + oversampling**: 0.8802 (data_01_r) - Best ensemble approach

### Performance Bottlenecks and Recommendations

#### Key Limitations Identified
1. **Word2Vec model dependency**: Full model required for optimal performance
2. **Neutral class ambiguity**: Hardest to distinguish from both negative and positive
3. **Class imbalance**: Positive class has lowest representation
4. **Feature sparsity**: TF-IDF features work well but lack semantic understanding

#### Practical Recommendations
1. **Production deployment**: Use SVM + TF-IDF(1,1) + OVO strategy
2. **Resource-constrained environments**: Use Naive Bayes + TF-IDF(1,1)
3. **When Word2Vec available**: Use SVM + Word2Vec for best accuracy
4. **For interpretability**: Use Random Forest + TF-IDF feature importance analysis

## Recommended Configurations

### 🏆 Production Environment Recommendation
- **Model**: SVM (RBF kernel)
- **Feature**: TF-IDF(1,1) or Word2Vec
- **Strategy**: OVO (one-vs-one)
- **Preprocessing**: Remove low-frequency words, no oversampling

### 🎯 Application Scenario Recommendations
- **High accuracy requirements**: SVM + Word2Vec (ROC-AUC: 0.89+)
- **Fast deployment**: Naive Bayes + TF-IDF (fast response)
- **Large-scale data**: Random Forest + OVR strategy (good scalability)

## Dataset Information

### Original Data
- **Source**: YouTube delivery robot related comments
- **Labels**: 0=negative, 1=positive, 2=neutral
- **Size**: Approximately 5,000 labeled comments

### Binary Classification Datasets

#### OVO Strategy Datasets
- `data_01.xlsx`: Negative vs Positive (2,298 samples)
- `data_02.xlsx`: Negative vs Neutral (2,668 samples)
- `data_12.xlsx`: Positive vs Neutral (1,910 samples)

#### OVR Strategy Datasets
- `data_0.xlsx`: Negative vs Others (3,393 samples)
- `data_1.xlsx`: Positive vs Others (3,343 samples)
- `data_2.xlsx`: Neutral vs Others (3,444 samples)

## Troubleshooting

### Common Issues and Solutions

1. **Word2Vec Download Issues**
   ```
   Error: Word2Vec model setup failed
   Solution: Check internet connection and try again
   Command: python main.py --setup-word2vec
   Alternative: Use TF-IDF features instead (remove --word2vec flag)
   ```

2. **Missing Data File**
   ```
   Error: File does not exist: data/combined_comments.xlsx
   Solution: Place combined_comments.xlsx in the data/ directory
   Command: python main.py --prepare
   ```

3. **Dependency Issues**
   ```
   Error: ModuleNotFoundError
   Solution: Install missing dependencies
   Command: uv pip install -r ../requirements.txt
   ```

4. **Path Reference Issues** (Fixed in latest version)
   ```
   Issue: Hardcoded paths in data_manager.py
   Status: Fixed - now uses config.py constants
   ```

### Performance Optimization Tips

1. **Memory optimization**: Use sparse matrices for TF-IDF features
2. **Speed optimization**: Pre-compute feature vectors, cache model results
3. **Accuracy optimization**: Ensemble multiple models, use voting or weighted strategies


## Technical Implementation Details

### Feature Engineering

#### TF-IDF Configuration
```python
# Optimal configuration
TfidfVectorizer(
    ngram_range=(1,1),      # 1-gram performs best
    max_features=10000,     # Avoid dimension explosion
    stop_words='english'    # Use custom stop words
)
```

#### Word2Vec Configuration
```python
# Using pre-trained model (with intelligent fallback)
Word2VecVectorizer(
    model=GoogleNews_model,  # Auto-fallback to virtual model
    bow='avg',              # Average word vectors
    shift_to_positive=False # Keep original vectors
)
```

### Data Preprocessing Pipeline
1. **Text cleaning**: Remove special characters, numbers
2. **Tokenization**: NLTK tokenizer
3. **Stop word filtering**: Custom stop word list
4. **Lemmatization**: WordNetLemmatizer
5. **Low-frequency word processing**: Optional low-frequency word filtering

### Model Evaluation Metrics
```python
# Main evaluation metrics
metrics = {
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'classification_report': classification_report(y_test, y_pred)
}
```

## Command Line Interface

```bash
# View all options
python main.py --help

# Model selection
python main.py --train --model svm --type ovo --ngram 2

# Feature selection
python main.py --train --word2vec --kfold

# Complete test
python main.py --binary-test comprehensive

# Fix Word2Vec encoding issues
python main.py --fix-word2vec
```

## Contribution Guidelines

### Code Standards
- Use Python 3.11+ (3.11.12 recommended)
- Follow PEP 8 coding standards
- Add detailed docstring documentation
- Include unit tests

### Submission Process
1. Fork the project repository
2. Create feature branch
3. Write test cases
4. Submit Pull Request

## License

This project uses the MIT License. See LICENSE file for details.

## Acknowledgments

- **Data source**: YouTube delivery robot comment dataset
- **Pre-trained model**: Google News Word2Vec model
- **Open source libraries**: scikit-learn, gensim, nltk, pandas

---

*Last updated: 2026-01-07*
*Version: v1.1.0*
*Maintainer: Project Development Team*