# Sentiment Classification Models

This document describes the sentiment classification models implemented in the YouTube-SC project, including both traditional machine learning approaches and deep learning architectures.

## Overview
The project implements a comprehensive sentiment analysis system with two main approaches:
1. **Machine Learning Models**: Traditional algorithms with feature engineering
2. **Deep Learning Models**: BERT-based transformer architecture with sequence modeling layers

Both approaches support ternary classification (positive, negative, neutral) and can be adapted for binary classification tasks.

## Machine Learning Sentiment Classification

### Available Models
| Model | Description | Key Features |
|-------|-------------|--------------|
| **Support Vector Machine (SVM)** | RBF kernel with parameter tuning | High accuracy, good generalization |
| **Naive Bayes** | Multinomial variant for text classification | Fast training, probabilistic outputs |
| **Random Forest** | Ensemble of decision trees | Feature importance, robust to overfitting |
| **Decision Tree** | Baseline interpretable model | Easy to understand, fast inference |

### Feature Engineering

#### TF-IDF Features
- **Term Frequency-Inverse Document Frequency** vectors
- **N-gram ranges**: Configurable (unigrams, bigrams, trigrams)
- **Vocabulary size**: Optimized for performance (typically 5000-10000 features)
- **Sublinear TF scaling**: Apply log transformation to term frequencies

#### Word Embeddings
- **Word2Vec**: Pre-trained on Google News corpus (300-dimensional vectors)
- **Fallback mechanism**: Skip words not in vocabulary, use average of known words
- **Alternative embeddings**: FastText or GloVe can be substituted
- **Vector aggregation**: Mean pooling of word vectors per document

### Classification Strategies

#### One-vs-One (OVO)
- Train binary classifiers for each pair of classes (3 classifiers for 3 classes)
- Final prediction by majority voting
- Works well when classes are balanced

#### One-vs-Rest (OVR)
- Train binary classifiers for each class vs all others (3 classifiers for 3 classes)
- Final prediction by highest confidence score
- More efficient for multi-class problems

### Performance Results

#### Optimal Configuration
- **Model**: SVM with RBF kernel
- **Features**: TF-IDF with (1,1) n-gram range (unigrams only)
- **Strategy**: One-vs-One (OVO)
- **ROC-AUC**: 0.8946 (binary classification)

#### Additional Results
- **Best ROC-AUC**: 0.8977 with optimized configurations
- **Cross-validation**: 5-fold stratified CV for reliable estimates
- **Class-wise metrics**: Precision, recall, F1 for each sentiment class

### Configuration Management
Each experiment configuration is managed through the `config.py` file in the `sentiment_classification_ML` module, allowing easy adjustment of:
- Model type (SVM, NB, RF, DT)
- Feature type (TF-IDF, Word2Vec)
- N-gram range for TF-IDF
- Classification strategy (OVO, OVR)
- Hyperparameter ranges for grid search

## Deep Learning Sentiment Classification (BERT-based)

### Model Architecture

#### Base Transformer
- **Pre-trained model**: `bert-base-uncased` (English BERT)
- **Layers**: 12 transformer layers, 768 hidden dimensions, 12 attention heads
- **Vocabulary**: 30,522 WordPiece tokens
- **Maximum sequence length**: 512 tokens (truncated/padded as needed)

#### Sequence Modeling Layers
- **Bidirectional LSTM**: 128 hidden units, 2 layers
- **Bidirectional GRU**: 128 hidden units, 2 layers
- **Dropout**: 0.3-0.5 between layers for regularization
- **Linear classifier**: Final dense layer with softmax activation for 3 classes

#### Architecture Diagram
```
Input Text → BERT Tokenizer → BERT Encoder → [BiLSTM + BiGRU] → Dropout → Linear → Softmax → Output
```

### Training Configuration

#### Optimizer
- **Algorithm**: AdamW with weight decay
- **Learning rate**: 2e-5 with linear warmup and decay
- **Batch size**: 32 for training, 32 for validation (configurable)
- **Epochs**: 10-20 with early stopping

#### Regularization
- **Dropout**: 0.3-0.5 on transformer outputs and sequence layers
- **Gradient clipping**: Max norm of 1.0 to prevent exploding gradients
- **Weight decay**: 0.01 for L2 regularization

#### Loss Function
- **Cross-entropy loss** for multi-class classification
- **Class weighting** to handle imbalanced datasets
- **Label smoothing** (optional) for better calibration

### Fine-tuning Strategy

#### Transfer Learning
- Start with pre-trained BERT weights from HuggingFace
- Freeze early transformer layers initially
- Gradually unfreeze layers during training
- Fine-tune only classifier layers first, then entire model

#### Training Schedule
1. **Warmup phase**: 10% of training steps with increasing learning rate
2. **Fine-tuning phase**: Main training with gradual layer unfreezing
3. **Evaluation phase**: Regular validation checks and early stopping

### Recent Improvements

#### Path Standardization
- Fixed hardcoded file paths to use configuration-based paths
- Improved cross-platform compatibility (Windows/Linux/Mac)

#### Label Mapping
- Standardized sentiment label mapping (0: negative, 1: positive, 2: neutral)
- Consistent label handling across all modules

#### AdamW Compatibility
- Updated to use modern AdamW implementation from transformers library
- Fixed weight decay parameter handling

### Performance Characteristics

#### Training Time
- **BERT fine-tuning**: ~2-4 hours on GPU (NVIDIA RTX 3080)
- **Inference**: ~1000 comments/second on GPU
- **Memory usage**: ~1.5GB VRAM for batch size 32

#### Accuracy Metrics
- **Validation accuracy**: ~85-90% on balanced datasets
- **Test accuracy**: Comparable to validation performance
- **Class-wise F1**: Balanced across all three sentiment classes

## Model Comparison

### Strengths and Weaknesses

| Model Type | Strengths | Weaknesses | Best For |
|------------|-----------|------------|----------|
| **SVM + TF-IDF** | High accuracy, interpretable features | Manual feature engineering, limited context | Production deployment, quick iterations |
| **BERT + LSTM/GRU** | Contextual understanding, state-of-the-art | Computational cost, longer training | Research, highest accuracy requirements |

### Use Case Recommendations

#### Quick Analysis
- Use **SVM with TF-IDF** for fast results
- Configure with OVO strategy and (1,1) n-grams
- Results available in minutes on standard hardware

#### Research/Publication
- Use **BERT with LSTM/GRU** for best accuracy
- Fine-tune with early stopping and learning rate scheduling
- Report multiple metrics (accuracy, F1, ROC-AUC)

#### Exploratory Analysis
- Try multiple models with different feature sets
- Compare OVO vs OVR strategies
- Analyze feature importance and misclassifications

## Integration with Other Modules

### Data Flow
```
Raw comments → Preprocessing → Feature extraction → Model training → Predictions
      ↓
Clustering analysis, Topic modeling, Text statistics
```

### Output Formats
- **Predictions**: CSV files with comment ID, true label, predicted label, confidence scores
- **Models**: Saved as `.pkl` files (ML) or PyTorch `.pt` files (BERT)
- **Reports**: Classification reports, confusion matrices, ROC curves

## Reproducibility

### Random Seeds
- Fixed random seeds for numpy, random, and torch
- Deterministic algorithms where possible
- Seed configuration in `config.py`

### Version Control
- Model architectures versioned in code repository
- Hyperparameter configurations documented
- Training scripts include commit hashes in output

### Experiment Tracking
- TensorBoard logs for deep learning experiments
- CSV logs for ML experiments
- Configuration snapshots with each trained model

## Future Extensions

### Planned Improvements
- **Multilingual BERT** for non-English comments
- **Ensemble methods** combining ML and deep learning predictions
- **Explainable AI** techniques for model interpretability
- **Real-time sentiment analysis** API

### Research Directions
- **Cross-domain adaptation** to other delivery robot contexts
- **Temporal sentiment analysis** tracking changes over time
- **Multimodal analysis** combining text with video metadata