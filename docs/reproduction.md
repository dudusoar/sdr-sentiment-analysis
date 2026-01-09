# Experiment Reproduction

This guide provides detailed instructions for reproducing the experimental results reported in the associated paper: *"Understanding Social Perceptions, Interactions, and Safety Aspects of Sidewalk Delivery Robots Using Sentiment Analysis"*.

## Prerequisites
Before starting, ensure you have:
1. **Environment Setup**: Complete the [Environment Setup](setup.md) guide
2. **Data Access**: The `data/combined_comments.xlsx` file in the correct location
3. **Activated Virtual Environment**: All commands assume you're in the project's virtual environment

## Quick Reproduction Script
For a complete reproduction of all experiments, run:
```bash
# From project root directory
python scripts/reproduce_all_experiments.py
```

This script will execute all experiments in sequence and generate a comprehensive report.

## Sentiment Classification Experiments

### Machine Learning Sentiment Classification

#### 1. Dataset Preparation
```bash
cd sentiment_classification_ML
python main.py --prepare
```
This command:
- Loads the raw comments from `data/combined_comments.xlsx`
- Applies text preprocessing (tokenization, stopword removal, lemmatization)
- Splits data into training (70%), validation (15%), and test (15%) sets
- Saves processed datasets to `data/comments/` directory

#### 2. Comprehensive Binary Classification Tests
```bash
python main.py --binary-test comprehensive
```
This runs a comprehensive evaluation of all model configurations:
- **Models**: SVM, Naive Bayes, Random Forest, Decision Tree
- **Features**: TF-IDF with n-gram ranges (1,1), (1,2), (1,3)
- **Strategies**: One-vs-One (OVO), One-vs-Rest (OVR)
- **Output**: CSV files with performance metrics in `results/dataset/`

#### 3. Specific Configuration Testing
To test the optimal configuration reported in the paper:
```bash
python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo --ngram 1
```
Expected results:
- **ROC-AUC**: ~0.8946
- **Accuracy**: Varies based on test split (typically 85-90%)
- **Output**: Detailed classification report and confusion matrix

#### 4. Validation Script
```bash
python test_binary_framework.py
```
This script validates the binary classification framework and reports key metrics.

### BERT Sentiment Classification

#### 1. Training BERT Model
```bash
cd ../sentiment_classification_Bert/code
python main.py --epoches 10 --train_batch_size 32 --valid_batch_size 32 --learning_rate 2e-5
```
Parameters:
- `--epoches`: Number of training epochs (10 recommended)
- `--train_batch_size`: Batch size for training (32 recommended)
- `--valid_batch_size`: Batch size for validation (32 recommended)
- `--learning_rate`: Learning rate for AdamW optimizer (2e-5 recommended)

#### 2. Evaluation
After training, the model automatically evaluates on the test set and reports:
- **Accuracy**: Overall classification accuracy
- **Per-class metrics**: Precision, recall, F1 for negative, positive, neutral
- **Confusion matrix**: Visual representation of predictions

#### 3. Using Pre-trained Model
To use the pre-trained model from the paper:
```bash
python predict.py --model_path models/bert_sentiment_final.pt --input_file data/test_comments.csv
```

## Topic Modeling Experiments

### 1. Running LDA Topic Modeling
```bash
cd ../topic_modeling
python topic_modeling_analysis.py
```
This script:
- Loads preprocessed comments
- Trains LDA model with 10 topics
- Calculates coherence scores (C_v and UMass)
- Generates topic visualizations
- Saves results to `results/` directory

### 2. Topic Evaluation
Expected results:
- **Number of topics**: 10
- **C_v coherence**: > 0.45 (higher is better)
- **Top keywords**: See `results/topic_keywords.csv`
- **Topic labels**: Manual interpretation based on paper findings

### 3. Visualization Generation
The script produces several visualizations:
- **Topic distance map**: `results/topic_distance.html` (interactive)
- **Word clouds**: `results/wordcloud_topic_X.png` for each topic
- **Topic prevalence**: `results/topic_distribution.png`

## Clustering Analysis

### 1. Running Clustering Analysis
```bash
cd ../sdr_clustering_analysis
python main.py
```
This performs:
- K-means clustering on comment embeddings
- Elbow method for optimal cluster count determination
- Cluster visualization using PCA/t-SNE
- Cluster interpretation based on frequent terms

### 2. Expected Output
- **Optimal clusters**: Typically 5-8 clusters emerge naturally
- **Cluster labels**: Saved in `results/cluster_assignments.csv`
- **Visualizations**: 2D scatter plots in `results/cluster_visualization.png`

## Text Statistics Analysis

### 1. Yearly Word Frequency Analysis
```bash
cd ../yearly_word_frequency
python main.py
```
This analysis:
- Calculates word frequencies by year
- Identifies trending terms over time
- Generates frequency plots and heatmaps

### 2. Output Files
- **Word frequency tables**: `results/yearly_frequencies.csv`
- **Trend visualizations**: `results/word_trends.png`
- **Statistical summaries**: `results/text_statistics_summary.csv`

## Complete Pipeline Reproduction

### Step-by-Step Reproduction
For a complete reproduction of all analyses in sequence:

```bash
# 1. Setup environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# 2. Prepare data
cd sentiment_classification_ML
python main.py --prepare

# 3. Run ML sentiment analysis
python main.py --binary-test comprehensive

# 4. Run BERT sentiment analysis
cd ../sentiment_classification_Bert/code
python main.py --epoches 10 --train_batch_size 32 --valid_batch_size 32

# 5. Run topic modeling
cd ../topic_modeling
python topic_modeling_analysis.py

# 6. Run clustering analysis
cd ../sdr_clustering_analysis
python main.py

# 7. Run text statistics
cd ../yearly_word_frequency
python main.py

# 8. Generate integrated report
cd ..
python scripts/generate_integrated_report.py
```

### Expected Timeline
- **Total time**: 4-8 hours depending on hardware
- **ML sentiment analysis**: 30-60 minutes
- **BERT training**: 2-4 hours (GPU recommended)
- **Topic modeling**: 30-60 minutes
- **Clustering & statistics**: 30 minutes each

## Reproducibility Measures

### Fixed Random Seeds
All experiments use fixed random seeds for reproducibility:
- **NumPy/Python random**: seed = 42
- **scikit-learn**: random_state = 42
- **PyTorch**: torch.manual_seed(42)
- **Gensim**: random_state = 42

### Configuration Files
Each module has a `config.py` file containing all experiment parameters:
- `sentiment_classification_ML/config.py`
- `sentiment_classification_Bert/code/config.py`
- `topic_modeling/config.py`
- `sdr_clustering_analysis/config.py`
- `yearly_word_frequency/config.py`

### Version Information
The exact versions used in the paper experiments:
- **Python**: 3.11.12
- **scikit-learn**: 1.8.0
- **PyTorch**: 2.9.1+cpu
- **Transformers**: 4.57.3
- **Gensim**: 4.4.0
- **NLTK**: 3.9.2

## Output Verification

### Expected File Structure
After running all experiments, you should have:
```
results/
├── sentiment_classification_ML/
│   ├── dataset/              # ML experiment results
│   └── models/               # Trained ML models
├── sentiment_classification_Bert/
│   ├── models/               # Trained BERT models
│   └── predictions/          # BERT predictions
├── topic_modeling/
│   ├── models/               # LDA models
│   ├── visualizations/       # Topic visualizations
│   └── topics/               # Topic assignments
├── sdr_clustering_analysis/
│   ├── clusters/             # Cluster assignments
│   └── visualizations/       # Cluster plots
└── yearly_word_frequency/
    ├── frequencies/          # Word frequency tables
    └── visualizations/       # Trend plots
```

### Key Metrics to Verify
1. **ML Sentiment ROC-AUC**: Should be ~0.8946 for optimal configuration
2. **BERT Accuracy**: Should be ~85-90% on test set
3. **Topic Coherence**: C_v > 0.45 for interpretable topics
4. **Cluster Quality**: Clear separation in visualization plots
5. **Statistical Trends**: Identifiable patterns in word frequencies over time

## Troubleshooting Reproduction

### Common Issues

#### 1. Missing Data Files
```bash
# Ensure combined_comments.xlsx is in data/ directory
ls data/combined_comments.xlsx

# If missing, contact project maintainers for dataset access
```

#### 2. Memory Issues
For large datasets or BERT training:
```bash
# Reduce batch sizes
python main.py --train_batch_size 16 --valid_batch_size 16

# Use gradient accumulation
python main.py --gradient_accumulation_steps 2
```

#### 3. GPU Availability
Check GPU availability for BERT training:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

#### 4. Package Version Conflicts
Recreate environment with exact versions:
```bash
uv pip install -r requirements.txt --force-reinstall
```

### Verification Script
Run the verification script to check reproduction success:
```bash
python scripts/verify_reproduction.py
```
This script checks:
- All required output files exist
- Key metrics match expected ranges
- Configurations match paper descriptions

## Customizing Experiments

### Modifying Parameters
To experiment with different configurations:

#### ML Sentiment Analysis
```bash
cd sentiment_classification_ML
python main.py --train --model RandomForest --feature Word2Vec --strategy ovr
```

#### BERT Training
```bash
cd sentiment_classification_Bert/code
python main.py --epoches 15 --learning_rate 3e-5 --weight_decay 0.01
```

#### Topic Modeling
Modify `topic_modeling/config.py`:
```python
NUM_TOPICS = 15  # Change from 10 to 15
PASSES = 20      # Increase training passes
```

### Adding New Analyses
The modular structure allows adding new analysis modules:
1. Create new directory with `main.py` and `config.py`
2. Follow existing patterns for data loading and output saving
3. Update main README with new module description

## Citation and Attribution
When reproducing these experiments for research purposes, please cite:

```bibtex
@article{du2025sdr,
  title   = {Understanding Social Perceptions, Interactions, and Safety Aspects of Sidewalk Delivery Robots Using Sentiment Analysis},
  author  = {Du, Yuchen and Le, Tho V.},
  journal = {Transportation Research Record},
  year    = {2025},
  doi     = {10.1177/03611981251394686}
}
```

## Support and Questions
For issues with reproduction:
1. Check [Troubleshooting Guide](troubleshooting.md)
2. Review [GitHub Issues](https://github.com/yourusername/Youtube-SC/issues)
3. Contact project maintainers with detailed error information

This reproduction guide ensures complete transparency and enables independent verification of all results reported in the paper.