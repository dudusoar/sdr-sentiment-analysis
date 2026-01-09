# Usage Examples

This document provides comprehensive usage examples for running each analysis module in the YouTube-SC project. All commands assume you have activated the virtual environment as described in the [Environment Setup](setup.md) guide.

## Quick Start

### Basic Workflow
```bash
# 1. Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 2. Prepare data (first time only)
cd sentiment_classification_ML
python main.py --prepare

# 3. Run desired analyses (examples below)
```

## Machine Learning Sentiment Classification

### Data Preparation
```bash
cd sentiment_classification_ML

# Prepare datasets (training, validation, test splits)
python main.py --prepare

# Check data preparation output
ls data/comments/
# Expected: train.csv, val.csv, test.csv
```

### Comprehensive Testing
Run all model configurations to find optimal settings:
```bash
python main.py --binary-test comprehensive
```
This tests:
- 4 models: SVM, Naive Bayes, Random Forest, Decision Tree
- 3 n-gram ranges: (1,1), (1,2), (1,3)
- 2 strategies: One-vs-One (OVO), One-vs-Rest (OVR)
- Outputs: CSV files in `results/dataset/` with performance metrics

### Specific Configuration Testing
Test a specific configuration (e.g., optimal SVM configuration):
```bash
python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo --ngram 1
```
Expected output includes:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- ROC-AUC score (~0.8946 for optimal configuration)

### Training Custom Models
Train a model with custom parameters:
```bash
# Train SVM with TF-IDF features and OVO strategy
python main.py --train --model svm --type ovo --ngram 1

# Train Random Forest with Word2Vec features
python main.py --train --model rf --feature word2vec

# Train with custom train/test split (80/20)
python main.py --train --model svm --type ovo --train-ratio 0.8
```

### Model Evaluation
```bash
# Evaluate trained model on test set
python main.py --evaluate --model-path results/models/svm_tfidf_ovo.pkl

# Generate detailed evaluation report
python main.py --evaluate --model-path results/models/svm_tfidf_ovo.pkl --output report.html
```

### Word2Vec Setup
If Word2Vec embeddings are not available:
```bash
# Download Google News Word2Vec embeddings (first time)
python main.py --setup-word2vec

# Alternative: Use TF-IDF features if download fails
python main.py --train --model svm --type ovo --ngram 1
```

## BERT Sentiment Classification

### Training BERT Model
```bash
cd sentiment_classification_Bert/code

# Basic training with default parameters
python main.py

# Training with custom parameters
python main.py --epoches 10 --train_batch_size 32 --valid_batch_size 32 --learning_rate 2e-5

# Training with early stopping
python main.py --epoches 20 --patience 3 --min_delta 0.001

# Training on GPU (if available)
python main.py --device cuda --epoches 10
```

### Parameter Options
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epoches` | Number of training epochs | 10 |
| `--train_batch_size` | Batch size for training | 32 |
| `--valid_batch_size` | Batch size for validation | 32 |
| `--learning_rate` | Learning rate for AdamW | 2e-5 |
| `--weight_decay` | Weight decay for regularization | 0.01 |
| `--warmup_steps` | Number of warmup steps | 100 |
| `--max_seq_len` | Maximum sequence length | 128 |
| `--device` | Device to use (cpu/cuda) | cpu |
| `--patience` | Early stopping patience | 5 |
| `--min_delta` | Minimum improvement for early stopping | 0.001 |

### Using Pre-trained Model
```bash
# Make predictions with trained model
python predict.py --model_path models/bert_sentiment_final.pt --input_file data/test_comments.csv

# Generate predictions with confidence scores
python predict.py --model_path models/bert_sentiment_final.pt --input_file data/test_comments.csv --output predictions.csv --include_confidence

# Interactive prediction
python predict.py --model_path models/bert_sentiment_final.pt --text "I love delivery robots, they're so convenient!"
```

### Model Evaluation
```bash
# Evaluate model on test set
python evaluate.py --model_path models/bert_sentiment_final.pt --test_file data/test.csv

# Generate detailed evaluation metrics
python evaluate.py --model_path models/bert_sentiment_final.pt --test_file data/test.csv --output evaluation_report.html
```

## Clustering Analysis

### Running Clustering
```bash
cd sdr_clustering_analysis

# Basic clustering with default parameters
python main.py

# Clustering with specific number of clusters
python main.py --n-clusters 8

# Clustering with different feature types
python main.py --features tfidf  # Use TF-IDF features
python main.py --features word2vec  # Use Word2Vec embeddings

# Save cluster assignments to custom file
python main.py --output my_clusters.csv
```

### Visualization Options
```bash
# Generate 2D visualization using PCA
python main.py --visualize pca --output clusters_pca.png

# Generate 2D visualization using t-SNE
python main.py --visualize tsne --output clusters_tsne.png

# Generate 3D visualization
python main.py --visualize pca --dimensions 3 --output clusters_3d.html
```

### Cluster Analysis
```bash
# Analyze cluster contents and generate report
python main.py --analyze --output cluster_report.txt

# Extract top terms for each cluster
python main.py --extract-terms --top-n 10 --output cluster_terms.csv
```

## Topic Modeling

### Running LDA Topic Modeling
```bash
cd topic_modeling

# Basic topic modeling with 10 topics
python topic_modeling_analysis.py

# Topic modeling with custom number of topics
python topic_modeling_analysis.py --num-topics 15

# Adjust LDA parameters
python topic_modeling_analysis.py --passes 20 --iterations 2000 --random-state 42
```

### Topic Analysis
```bash
# Generate topic coherence report
python topic_modeling_analysis.py --coherence --output coherence_report.csv

# Extract topic keywords
python topic_modeling_analysis.py --extract-keywords --top-n 15 --output topic_keywords.csv

# Assign topics to documents
python topic_modeling_analysis.py --assign-topics --output document_topics.csv
```

### Visualization
```bash
# Generate interactive topic visualization
python topic_modeling_analysis.py --visualize --output topic_viz.html

# Create word clouds for each topic
python topic_modeling_analysis.py --wordclouds --output-dir wordclouds/

# Generate topic prevalence plot
python topic_modeling_analysis.py --prevalence-plot --output topic_prevalence.png
```

## Text Statistics

### Word Frequency Analysis
```bash
cd yearly_word_frequency

# Basic yearly word frequency analysis
python main.py

# Analyze specific year range
python main.py --start-year 2020 --end-year 2023

# Focus on specific terms
python main.py --focus-terms robot,delivery,autonomous --output focus_analysis.csv
```

### Trend Analysis
```bash
# Identify trending terms over time
python main.py --trend-analysis --output trends.csv

# Generate trend visualization
python main.py --trend-plot --output trend_plot.png

# Calculate term frequency changes
python main.py --frequency-changes --output frequency_changes.csv
```

### Statistical Reports
```bash
# Generate comprehensive text statistics report
python main.py --full-report --output text_statistics_report.html

# Calculate readability scores
python main.py --readability --output readability_scores.csv

# Analyze sentiment distribution over time
python main.py --sentiment-trends --output sentiment_trends.png
```

## Integrated Analysis Pipeline

### Complete Analysis Workflow
Run the complete analysis pipeline end-to-end:

```bash
#!/bin/bash
# complete_analysis.sh

echo "Step 1: Data Preparation"
cd sentiment_classification_ML
python main.py --prepare

echo "Step 2: ML Sentiment Analysis"
python main.py --binary-test comprehensive

echo "Step 3: BERT Sentiment Analysis"
cd ../sentiment_classification_Bert/code
python main.py --epoches 10 --train_batch_size 32

echo "Step 4: Clustering Analysis"
cd ../sdr_clustering_analysis
python main.py

echo "Step 5: Topic Modeling"
cd ../topic_modeling
python topic_modeling_analysis.py

echo "Step 6: Text Statistics"
cd ../yearly_word_frequency
python main.py

echo "Analysis complete! Results saved to respective results/ directories."
```

### Custom Pipeline Configuration
Create a custom pipeline script:

```python
# custom_pipeline.py
import subprocess
import sys

def run_analysis():
    steps = [
        ("Data Preparation", "cd sentiment_classification_ML && python main.py --prepare"),
        ("ML Sentiment", "cd sentiment_classification_ML && python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo"),
        ("BERT Training", "cd sentiment_classification_Bert/code && python main.py --epoches 5"),
        ("Clustering", "cd sdr_clustering_analysis && python main.py --n-clusters 6"),
        ("Topic Modeling", "cd topic_modeling && python topic_modeling_analysis.py --num-topics 8"),
    ]

    for name, cmd in steps:
        print(f"Running: {name}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in {name}: {result.stderr}")
            sys.exit(1)
        print(f"Completed: {name}\n")

    print("All analyses completed successfully!")

if __name__ == "__main__":
    run_analysis()
```

## Testing and Validation

### Module Testing
```bash
# Test ML sentiment classification framework
cd sentiment_classification_ML
python test_binary_framework.py

# Test BERT model loading and inference
cd ../sentiment_classification_Bert/code
python test_bert_model.py

# Test clustering algorithms
cd ../sdr_clustering_analysis
python test_clustering.py

# Test topic modeling coherence
cd ../topic_modeling
python test_lda_coherence.py
```

### Configuration Testing
```bash
# Test all module configurations
cd scripts
python test_all_configs.py

# Validate data integrity
python validate_data.py --data-dir ../data

# Check environment setup
python check_environment.py
```

## Advanced Usage

### Batch Processing
```bash
# Process multiple configurations in batch
cd sentiment_classification_ML
python batch_processor.py --config-file batch_configs.json

# Parallel processing (requires GNU parallel)
find . -name "*.py" -type f | parallel python {}
```

### Custom Feature Engineering
```bash
# Create custom TF-IDF features
cd sentiment_classification_ML
python custom_features.py --ngram-range 1 3 --max-features 10000 --output custom_features.pkl

# Train with custom features
python main.py --train --model svm --feature-file custom_features.pkl
```

### Cross-Validation
```bash
# Run k-fold cross-validation
cd sentiment_classification_ML
python cross_validate.py --model svm --folds 5 --feature TF-IDF --strategy ovo

# Nested cross-validation for hyperparameter tuning
python nested_cv.py --model svm --outer-folds 5 --inner-folds 3
```

## Troubleshooting Common Issues

### Virtual Environment Issues
```bash
# Check if virtual environment is activated
echo $VIRTUAL_ENV  # Should show path to .venv

# Reactivate if needed
deactivate
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Package Import Errors
```bash
# Reinstall requirements
uv pip install -r requirements.txt --force-reinstall

# Check specific package
python -c "import pandas; print(pandas.__version__)"
```

### Memory Issues
```bash
# Reduce batch sizes for BERT training
python main.py --train_batch_size 16 --valid_batch_size 16

# Use gradient accumulation
python main.py --gradient_accumulation_steps 2

# Process data in chunks
python main.py --chunk-size 1000
```

## Getting Help

### Command Help
```bash
# Get help for any module
cd sentiment_classification_ML
python main.py --help

cd ../sentiment_classification_Bert/code
python main.py --help

# Check module-specific documentation
cat README.md  # In each module directory
```

### Debug Mode
```bash
# Run with verbose output
python main.py --verbose --debug

# Log to file
python main.py --log-file analysis.log --log-level DEBUG
```

For additional help, refer to:
- [Troubleshooting Guide](troubleshooting.md)
- [Module-specific README files](../README.md)
- [GitHub Issues](https://github.com/yourusername/Youtube-SC/issues)

This usage guide covers the most common scenarios. For advanced use cases or custom analyses, refer to the source code and configuration files in each module.