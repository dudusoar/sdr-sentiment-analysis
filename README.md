# YouTube-SC: YouTube Sentiment and Clustering Analysis

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📋 Project Overview

YouTube-SC is a comprehensive analysis toolkit designed for extracting insights from YouTube delivery robot comments. The project employs multiple machine learning and natural language processing techniques to perform sentiment analysis, clustering, topic modeling, and text statistics on user comments related to delivery robots.

**Primary Objective**: Analyze YouTube delivery robot comments through:
- **Sentiment Classification** (ML-based and Deep Learning-based)
- **Clustering Analysis** to discover comment patterns
- **Topic Modeling** to extract discussion themes
- **Text Statistics** for quantitative analysis

## ✨ Key Features

- **Multi-Method Sentiment Analysis**: Both traditional ML models (SVM, Naive Bayes, Random Forest) and deep learning (BERT) approaches
- **Comprehensive Clustering**: K-means and other clustering algorithms to group similar comments
- **Topic Modeling**: LDA-based topic extraction to identify discussion themes
- **Text Statistics**: Word frequency analysis, yearly trends, and text metrics
- **Modular Architecture**: Independent modules for each analysis type
- **Production-Ready Configurations**: Optimized model configurations for different use cases
- **Custom Management Tools**: Built-in task tracking, bug logging, and environment management skills

## 🏗️ Project Structure

```
Youtube-SC/
├── sentiment_classification_ML/      # Machine learning sentiment classification
│   ├── main.py                      # Entry point for ML sentiment analysis
│   ├── config.py                    # Configuration management
│   ├── src/                         # Core source code
│   └── README.md                    # Module documentation
├── sentiment_classification_Bert/    # BERT-based sentiment classification
│   └── code/                        # BERT implementation with LSTM/GRU layers
├── sdr_clustering_analysis/         # Comment clustering analysis
│   └── main.py                      # Clustering entry point
├── topic_modeling/                  # Topic modeling analysis
│   └── topic_modeling_analysis.py   # Topic modeling entry point
├── yearly_word_frequency/           # Yearly word frequency analysis
├── data/                            # Shared data directory
│   ├── combined_comments.xlsx       # Comment dataset file
│   └── Video_statistics/            # Video statistics
├── requirements.txt                 # Project dependencies (unified)
├── setup-environment.sh             # Linux/Mac environment setup
├── setup-environment.bat            # Windows environment setup
├── .claude/                         # Claude Code configuration
│   ├── task-board.md                # Project task management
│   ├── bug-log.md                   # Bug tracking and debugging
│   └── skills/                      # Custom management skills
```

## 🛠️ Technology Stack

- **Python**: 3.11
- **Machine Learning**: scikit-learn, gensim, nltk
- **Deep Learning**: PyTorch, Transformers, Sentence-Transformers
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud
- **Environment Management**: uv (for virtual environments)

## 🚀 Quick Start

### Environment Overview

- **Python Version**: 3.11.12
- **Virtual Environment Path**: `.venv/`
- **Dependency Management Tool**: uv
- **Number of Installed Packages**: 138
- **Environment Creation Date**: 2026-01-06

### Prerequisites
- Python 3.11+ (3.11.12 recommended for optimal compatibility)
- [uv](https://github.com/astral-sh/uv)
- Git

### Installation

**Option 1: Using uv (Recommended)**
```bash
# Clone the repository
git clone https://github.com/yourusername/Youtube-SC.git
cd Youtube-SC

# Set up environment using automated scripts
# Windows
setup-environment.bat

# Linux/Mac
./setup-environment.sh
```


### Activating Virtual Environment

After installation, you need to activate the virtual environment before running any project code:

**Windows (PowerShell/CMD):**
```cmd
.venv\Scripts\activate
```

**Windows (Git Bash):**
```bash
source .venv/Scripts/activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Verification Command:** After activation, you should see `(.venv)` prefix in your terminal prompt.

### Verify Installation
```bash
# After activating environment, verify key packages
python -c "import pandas; print(f'✅ pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'✅ scikit-learn: {sklearn.__version__}')"
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
```

### Managing Dependencies

Once the environment is set up, you can manage dependencies using the following commands:

**View installed packages:**
```bash
uv pip list
```

**Update packages:**
```bash
uv update
# or update specific package
uv update package_name
```

**Add new packages:**
```bash
uv add package_name
```

**Install from requirements.txt:**
```bash
uv pip install -r requirements.txt
```

**Key Package Versions (Current Environment):**
- pandas: 2.3.3
- numpy: 2.4.0
- scikit-learn: 1.8.0
- torch: 2.9.1+cpu
- transformers: 4.57.3
- gensim: 4.4.0
- nltk: 3.9.2
- matplotlib: 3.10.8
- seaborn: 0.13.2
- wordcloud: 1.9.5

## 📊 Module Descriptions

### 1. Machine Learning Sentiment Classification (`sentiment_classification_ML/`)
**Status**: ✅ Production Ready

Comprehensive sentiment analysis system using traditional ML algorithms:
- **Models**: SVM, Naive Bayes, Random Forest, Decision Tree
- **Strategies**: One-vs-One (OVO) and One-vs-Rest (OVR) classification
- **Features**: TF-IDF, Word2Vec (with intelligent fallback)
- **Performance**: ROC-AUC up to 0.8977 with optimal configurations

**Optimal Configuration**: SVM + TF-IDF(1,1) + OVO strategy (ROC-AUC: 0.8946)

### 2. BERT Sentiment Classification (`sentiment_classification_Bert/`)
**Status**: ✅ Available

Deep learning-based sentiment classification using BERT with LSTM/GRU layers:
- **Model**: Uses `bert-base-uncased` (English BERT model) for English YouTube comments
- **Architecture**: BERT encoder with bidirectional LSTM/GRU layers for sequence modeling
- **Classification**: Three-class sentiment analysis (positive, negative, neutral)
- **Recent Fixes**: Fixed hardcoded paths, standardized label mapping, updated AdamW compatibility
- **Training**: Supports configurable training epochs, batch sizes, and early stopping
- **Note**: Uses pre-trained HuggingFace transformers with custom loss functions

### 3. Clustering Analysis (`sdr_clustering_analysis/`)
**Status**: ✅ Production Ready

Unsupervised learning to discover comment patterns:
- **Algorithms**: K-means clustering
- **Applications**: Group similar comments, identify discussion clusters
- **Visualization**: Cluster visualization and analysis

### 4. Topic Modeling (`topic_modeling/`)
**Status**: ✅ Production Ready

Latent Dirichlet Allocation (LDA) for theme extraction:
- **Purpose**: Discover discussion topics within comments
- **Output**: Topic distributions and key terms
- **Analysis**: Topic coherence and visualization

### 5. Text Statistics (`yearly_word_frequency/`)
**Status**: ✅ Production Ready

Quantitative analysis of text data:
- **Features**: Word frequency analysis, yearly trends
- **Metrics**: Statistical measures of text characteristics
- **Visualization**: Frequency plots and trend analysis

## 🔄 Data Flow

```
Raw Comments (data/combined_comments.xlsx)
        ↓
    [Processing]
        ↓
┌─────────────────────────────┐
│  sentiment_classification_ML │ → Sentiment labels & analysis
│  sentiment_classification_Bert│ → Deep learning sentiment analysis
│  sdr_clustering_analysis    │ → Comment clusters
│  topic_modeling             │ → Discussion topics
│  yearly_word_frequency      │ → Statistical metrics
└─────────────────────────────┘
        ↓
Results saved to respective module's `results/` directories
```

## 💻 Usage Examples

### Running Individual Modules

```bash
# Activate environment first
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# ML Sentiment Analysis
cd sentiment_classification_ML
python main.py --help
python main.py --prepare              # Prepare datasets
python main.py --binary-test comprehensive  # Run comprehensive tests

# Clustering Analysis
cd ../sdr_clustering_analysis
python main.py

# BERT Sentiment Analysis
cd ../sentiment_classification_Bert/code
python main.py --help
python main.py --epoches 10 --train_batch_size 32 --valid_batch_size 32

# Topic Modeling
cd ../topic_modeling
python topic_modeling_analysis.py
```

### Testing Configuration

```bash
# Test specific ML configuration
cd sentiment_classification_ML
python main.py --binary-test specific --test-model SVM --test-feature TF-IDF --test-strategy ovo

# Run validation script
python test_binary_framework.py
```


## 🛠️ Project Management Tools

This project includes custom Claude Code skills for efficient management:

### Available Skills
1. **update-task-board**: Task management using markdown files
2. **log-debug-issue**: Bug and issue tracking system
3. **manage-python-env**: Python virtual environment management using uv

### Usage
```bash
# Update task board
claude /update-task-board

# Log debugging issues
claude /log-debug-issue

# Manage Python environment
claude /manage-python-env
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**:
   ```bash
   # Ensure virtual environment is activated
   source .venv/bin/activate

   # Reinstall dependencies
   uv pip install -r requirements.txt
   ```

2. **Missing data files**:
   - Place `combined_comments.xlsx` in `data/` directory
   - Run `python main.py --prepare` in sentiment_classification_ML module

3. **Word2Vec download issues**:
   ```bash
   # Use TF-IDF features instead
   python main.py --train --model svm --type ovo --ngram 1

   # Or try alternative download
   python main.py --setup-word2vec
   ```

4. **Encoding problems** (Chinese text):
   ```python
   # Add to Python scripts
   import sys
   import io
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
   ```

### Environment-Specific Issues

5. **Virtual environment activation failure**:
   - Check if `.venv` directory exists in project root
   - Recreate environment using `setup-environment.bat` (Windows) or `./setup-environment.sh` (Linux/Mac) if missing

6. **Package import errors**:
   - Ensure virtual environment is activated (should see `(.venv)` in terminal prompt)
   - Reactivate environment if needed
   - Reinstall packages: `uv pip install -r requirements.txt`

7. **Missing packages**:
   - Reinstall all dependencies: `uv pip install -r requirements.txt`
   - Check network connection if using online installation
   - For offline setup, ensure all required packages are available locally

### Important Notes

1. **Always activate virtual environment** before running any project code
2. **All project modules should run within activated environment**
3. **To recreate environment**: Delete `.venv` directory and run `setup-environment.bat` (Windows) or `./setup-environment.sh` (Linux/Mac)
4. **For encoding issues** (GBK errors on Windows): Use UTF-8 encoding for files or add encoding wrapper in Python scripts

### Development Recommendations

1. **Using Jupyter Notebook in virtual environment**:
   ```bash
   # Activate environment first
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows

   # Launch Jupyter
   jupyter notebook
   ```

2. **Installing development tools**:
   ```bash
   uv add --dev black flake8 pytest
   ```

3. **Regular package updates**:
   ```bash
   # Check for outdated packages
   uv update --outdated
   # Update all packages
   uv update
   ```

### Performance Optimization
- **Memory**: Use sparse matrices for TF-IDF features
- **Speed**: Pre-compute feature vectors, cache model results
- **Accuracy**: Ensemble multiple models, use voting strategies

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Code Standards
- Python 3.11+ compatibility (3.11.12 recommended)
- PEP 8 coding standards
- Detailed docstrings (Chinese with English translations)
- Unit tests for new functionality

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Module-Specific Guidelines
1. **New Modules**: Include `config.py` for configuration management
2. **Dependencies**: Update `requirements.txt` when adding packages
3. **Documentation**: Update module README and main project documentation
4. **Testing**: Include test scripts with assertions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: YouTube delivery robot comment dataset
- **Pre-trained Models**: Google News Word2Vec, BERT base models
- **Open Source Libraries**: scikit-learn, gensim, nltk, pandas, PyTorch, Transformers
- **Development Tools**: Claude Code, Serena AI for project management

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Youtube-SC/issues)
- **Documentation**: Check `.claude/` directory for detailed guides
- **Troubleshooting**: Review `.claude/bug-log.md` for common solutions

---

**Last Updated**: 2026-01-07
**Version**: v1.2.0
**Maintainer**: Project Development Team