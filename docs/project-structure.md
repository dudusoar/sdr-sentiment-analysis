# Project Structure

This document provides a detailed overview of the YouTube-SC repository structure, explaining the purpose and contents of each directory and file.

## Repository Overview
```
Youtube-SC/
├── sentiment_classification_ML/      # Machine learning sentiment classification
├── sentiment_classification_Bert/    # BERT-based sentiment classification
├── sdr_clustering_analysis/          # Comment clustering analysis
├── topic_modeling/                   # Topic modeling analysis
├── yearly_word_frequency/            # Yearly word frequency analysis
├── data/                             # Shared data directory
├── docs/                             # Project documentation
├── .claude/                          # Claude Code configuration
├── requirements.txt                  # Project dependencies
├── setup-environment.sh              # Linux/Mac environment setup
├── setup-environment.bat             # Windows environment setup
└── README.md                         # Main project documentation
```

## Core Analysis Modules

### 1. Machine Learning Sentiment Classification (`sentiment_classification_ML/`)

```
sentiment_classification_ML/
├── main.py                      # Entry point for ML sentiment analysis
├── config.py                    # Configuration management
├── src/                         # Core source code
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── feature_extractor.py     # TF-IDF and Word2Vec feature extraction
│   ├── models.py                # ML model implementations (SVM, NB, RF, DT)
│   ├── trainer.py               # Model training and evaluation
│   └── utils.py                 # Utility functions
├── data/                        # Module-specific data
│   ├── comments/                # Processed comment datasets
│   └── features/                # Pre-computed feature files
├── results/                     # Analysis outputs
│   ├── dataset/                 # Experiment results by configuration
│   ├── models/                  # Trained model files (.pkl)
│   └── reports/                 # Classification reports and visualizations
└── README.md                    # Module documentation
```

**Key Files**:
- `main.py`: Command-line interface with options for data preparation, training, and testing
- `config.py`: Centralized configuration for models, features, and paths
- `test_binary_framework.py`: Validation script for binary classification framework

### 2. BERT Sentiment Classification (`sentiment_classification_Bert/`)
```
sentiment_classification_Bert/
└── code/                        # BERT implementation with LSTM/GRU layers
    ├── main.py                  # Training and evaluation script
    ├── config.py                # BERT-specific configuration
    ├── src/
    │   ├── data_loader.py       # BERT tokenization and data preparation
    │   ├── model.py             # BERT + LSTM/GRU architecture
    │   ├── trainer.py           # Training loop with early stopping
    │   └── evaluator.py         # Model evaluation and metrics
    ├── data/                    # BERT training data
    │   ├── train.csv            # Training split
    │   ├── val.csv              # Validation split
    │   └── test.csv             # Test split
    ├── models/                  # Saved model checkpoints
    │   └── bert_sentiment_final.pt  # Pre-trained model
    ├── results/                 # Predictions and evaluations
    │   ├── predictions/         # Model predictions
    │   └── metrics/             # Performance metrics
    └── README.md                # BERT module documentation
```

**Key Features**:
- Uses `bert-base-uncased` (English BERT model) for English YouTube comments
- Architecture: BERT encoder with bidirectional LSTM/GRU layers for sequence modeling
- Three-class sentiment analysis (positive, negative, neutral)
- Configurable training epochs, batch sizes, and early stopping

### 3. Clustering Analysis (`sdr_clustering_analysis/`)

```
sdr_clustering_analysis/
├── main.py                      # Clustering entry point
├── config.py                    # Clustering configuration
├── src/
│   ├── data_loader.py           # Load preprocessed comments
│   ├── clustering.py            # K-means and other clustering algorithms
│   ├── visualizer.py            # Cluster visualization (PCA/t-SNE)
│   └── analyzer.py              # Cluster interpretation and analysis
├── data/                        # Clustering inputs
│   └── embeddings/              # Word embeddings for clustering
├── results/
│   ├── clusters/                # Cluster assignments
│   ├── visualizations/          # 2D/3D cluster plots
│   └── analysis/                # Cluster interpretation reports
└── README.md                    # Clustering documentation
```

**Applications**:
- Group similar comments using K-means clustering
- Identify natural discussion clusters beyond sentiment
- Visualize comment relationships in reduced dimensions

### 4. Topic Modeling (`topic_modeling/`)

```
topic_modeling/
├── topic_modeling_analysis.py   # Topic modeling entry point
├── config.py                    # LDA configuration
├── src/
│   ├── data_preprocessor.py     # Text preparation for topic modeling
│   ├── lda_model.py             # LDA implementation using gensim
│   ├── evaluator.py             # Coherence and perplexity evaluation
│   ├── visualizer.py            # Topic visualization (PyLDAVis)
│   └── interpreter.py           # Topic labeling and interpretation
├── data/                        # Processed text for topic modeling
│   └── corpus/                  # Document-term matrix
├── results/
│   ├── models/                  # Trained LDA models
│   ├── topics/                  # Topic assignments and keywords
│   ├── visualizations/          # Topic visualizations
│   └── reports/                 # Topic analysis reports
└── README.md                    # Topic modeling documentation
```

**Purpose**:
- Discover discussion topics within comments using Latent Dirichlet Allocation (LDA)
- Extract topic distributions and key terms
- Analyze topic coherence and visualization

### 5. Text Statistics (`yearly_word_frequency/`)

```
yearly_word_frequency/
├── main.py                      # Text statistics entry point
├── config.py                    # Statistics configuration
├── src/
│   ├── data_loader.py           # Load temporal comment data
│   ├── frequency_analyzer.py    # Word frequency calculations
│   ├── trend_analyzer.py        # Temporal trend analysis
│   └── visualizer.py            # Statistical visualizations
├── data/                        # Yearly comment data
│   └── yearly_comments/         # Comments organized by year
├── results/
│   ├── frequencies/             # Word frequency tables
│   ├── trends/                  # Temporal trend analysis
│   ├── visualizations/          # Frequency plots and heatmaps
│   └── statistics/              # Text statistical metrics
└── README.md                    # Text statistics documentation
```

**Features**:
- Word frequency analysis by year
- Statistical measures of text characteristics
- Visualization of frequency plots and trend analysis

## Shared Directories

### Data Directory (`data/`)
Central location for all project data:

```
data/
├── combined_comments.xlsx       # Main comment dataset file
├── Video_statistics/            # Video statistics data
│   ├── video_metadata.csv       # Video metadata
│   ├── engagement_stats.csv     # View counts, likes, etc.
│   └── temporal_data.csv        # Time-series data
├── processed/                   # Processed data from modules
│   ├── preprocessed_comments.csv # Cleaned and tokenized comments
│   ├── sentiment_labels.csv     # Manual sentiment annotations
│   └── temporal_splits/         # Data splits by time period
└── external/                    # External data resources
    ├── word2vec/                # Word2Vec model files
    └── stopwords/               Custom stopword lists
```

**Data Flow**:
- Raw data: `combined_comments.xlsx` (original dataset)
- Processed data: Intermediate files created by analysis modules
- Results: Outputs saved to module-specific `results/` directories

### Documentation Directory (`docs/`)
Project documentation in Markdown format:

```
docs/
├── setup.md                     # Environment setup instructions
├── data.md                      # Dataset description and structure
├── pipeline.md                  # Analysis pipeline overview
├── sentiment_models.md          # Sentiment model architectures
├── topic_modeling.md            # Topic modeling implementation
├── reproduction.md              # Experiment reproduction guide
├── project-structure.md         # This file - repository structure
├── usage.md                     # Usage examples and commands
├── troubleshooting.md           # Common issues and solutions
├── management-tools.md          # Custom Claude Code skills
└── development.md               # Development recommendations
```

## Configuration Management

### Module Configuration Files
Each module has its own `config.py` file for managing:
- **Data paths**: Input and output file locations
- **Model parameters**: Hyperparameters for algorithms
- **Feature settings**: TF-IDF n-grams, embedding dimensions, etc.
- **Experiment settings**: Random seeds, batch sizes, epochs

### Centralized Configuration
The project follows these configuration principles:
1. **Separation of configuration from code**: All tunable parameters in `config.py`
2. **Environment-specific settings**: Different configurations for development vs production
3. **Version control**: Configuration files tracked in git for reproducibility
4. **Documentation**: Each configuration parameter documented in comments

## Development Tools

### Claude Code Configuration (`.claude/`)
Custom tools for project management:

```
.claude/
├── task-board.md                # Project task management
├── bug-log.md                   # Bug tracking and debugging
├── SKILLS_README.md             # Skills documentation
├── CLAUDE.md                    # Project guide
└── skills/                      # Custom management skills
    ├── update-task-board/       # Task management skill
    ├── log-debug-issue/         # Bug logging skill
    └── manage-python-env/       # Python environment management skill
```

### Environment Setup Scripts
- `setup-environment.bat`: Windows environment setup script
- `setup-environment.sh`: Linux/Mac environment setup script
- `requirements.txt`: Unified project dependencies

## File Naming Conventions

### Python Files
- **Modules**: `snake_case.py` (e.g., `data_loader.py`, `feature_extractor.py`)
- **Main scripts**: `main.py` or descriptive names like `topic_modeling_analysis.py`
- **Test files**: `test_*.py` or `*_test.py`

### Data Files
- **Raw data**: Descriptive names with extensions (e.g., `combined_comments.xlsx`)
- **Processed data**: Indicate processing stage (e.g., `preprocessed_comments.csv`)
- **Model files**: Include model type and date (e.g., `svm_tfidf_20250107.pkl`)

### Output Files
- **Results**: Organized by module in `results/` directories
- **Reports**: Include experiment configuration in filename
- **Visualizations**: Descriptive names indicating content (e.g., `cluster_visualization.png`)

## Module Independence and Integration

### Independent Operation
Each analysis module can operate independently:
- Own data loading and preprocessing
- Self-contained configuration
- Separate results directory
- Individual entry point (`main.py`)

### Integration Points
Modules integrate through:
1. **Shared data**: Common dataset in `data/combined_comments.xlsx`
2. **Common preprocessing**: Consistent text cleaning pipeline
3. **Complementary analyses**: Sentiment, clustering, topic modeling provide different perspectives
4. **Unified output format**: CSV files with consistent column naming

### Data Flow Between Modules
```
Raw Comments
    ↓
[Preprocessing] → sentiment_classification_ML → Sentiment labels
    ↓
sentiment_classification_Bert → Deep learning predictions
    ↓
sdr_clustering_analysis → Comment clusters
    ↓
topic_modeling → Discussion topics
    ↓
yearly_word_frequency → Statistical trends
```

## Best Practices for Development

### Adding New Modules
When adding new analysis modules:
1. Create new directory with `main.py`, `config.py`, and `README.md`
2. Follow existing structure: `src/`, `data/`, `results/` subdirectories
3. Update `requirements.txt` with new dependencies
4. Add module description to main `README.md` and `docs/project-structure.md`

### Modifying Existing Modules
1. Update `config.py` for parameter changes
2. Document changes in module `README.md`
3. Consider backward compatibility for existing results
4. Update documentation in `docs/` directory as needed

### Data Management
1. Raw data in `data/` directory (not version controlled if large)
2. Processed data in module-specific `data/` directories
3. Results in module `results/` directories
4. Use relative paths in configuration files

This structure supports reproducible research, modular development, and clear organization of the YouTube-SC project.