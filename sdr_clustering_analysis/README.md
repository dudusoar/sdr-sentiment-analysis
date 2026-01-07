# SDR YouTube Comment Clustering Analysis

## Project Overview
This project performs automated clustering analysis on YouTube comments to uncover natural groupings and understand comment patterns. It combines SBERT text embeddings with K-means clustering, supports multiple K values analysis, and produces detailed cluster profiles with sentiment distribution and traceable indexing.

## Key Features
- **SBERT Text Embeddings**: Uses multilingual Sentence-BERT for feature extraction
- **Multi-K Clustering**: K-means clustering with K values from 3 to 7
- **Sentiment Integration**: Cross-analysis between clusters and manual sentiment labels
- **Cluster Profiling**: Keyword extraction and sample comments for each cluster
- **Visualization**: Cluster size distribution, sentiment distribution, and silhouette analysis
- **Traceability**: All output files maintain original comment IDs for easy data tracing

## Project Structure
```
sdr_clustering_analysis/
├── main.py                    # Main entry point for clustering pipeline
├── config.py                  # Configuration parameters and paths
├── requirements.txt           # Project dependencies
├── README.md                  # This documentation file
├── data/                      # Data directory
│   ├── combined_comments.xlsx      # Raw comment data
│   └── sbert_embeddings_sdr.pkl    # Cached sentence embeddings
├── src/                       # Source code modules
│   ├── data_loader.py         # Data loading and validation
│   ├── text_preprocessor.py   # Optional text preprocessing
│   ├── feature_extractor.py   # SBERT feature extraction
│   ├── clustering.py          # K-means clustering implementation
│   ├── evaluation.py          # Cluster evaluation and analysis
│   └── utils.py               # Utility functions
└── results/                   # Output directory
    ├── cluster_sentiment_distribution_summary.xlsx  # Summary file
    ├── k3_results/            # Clustering results for K=3
    ├── k4_results/            # Clustering results for K=4
    ├── k5_results/            # Clustering results for K=5
    ├── k6_results/            # Clustering results for K=6
    └── k7_results/            # Clustering results for K=7
```

## Modules Description

### `data_loader.py`
- Loads raw comment data from Excel files
- Validates required columns and ensures unique comment IDs
- Removes empty text entries

### `text_preprocessor.py`
- Performs optional lightweight text preprocessing
- Handles lowercase conversion, URL removal, and whitespace normalization

### `feature_extractor.py`
- Generates sentence embeddings using SBERT models
- Supports embedding caching to avoid recomputation
- Automatically detects GPU/CPU availability

### `clustering.py`
- Implements K-means clustering with optimal K selection
- Evaluates clustering quality using silhouette score, Calinski-Harabasz, and Davies-Bouldin metrics
- Supports testing multiple K values (3-7)

### `evaluation.py`
- Extracts top keywords for each cluster using TF-IDF
- Samples representative comments from each cluster
- Analyzes sentiment distribution across clusters
- Generates cluster profiles and cross-tabulation tables

### `utils.py`
- Utility functions for pickle serialization and CSV saving
- Device detection for SBERT models

## Analysis Pipeline
1. **Data Loading**: Load raw comments with validation
2. **Feature Extraction**: Generate SBERT sentence embeddings (with caching)
3. **Clustering**: Apply K-means clustering for K values 3-7
4. **Evaluation**:
   - Calculate internal clustering metrics
   - Extract keywords and sample comments for each cluster
   - Analyze sentiment distribution across clusters
5. **Output Generation**:
   - Save cluster assignments with original comment IDs
   - Generate cluster profiles (keywords and comments)
   - Create visualization plots
   - Export cross-tabulation tables

## Output Files
Each `kX_results` directory contains:
- `cluster_assignments_kX.csv` - Cluster labels for each comment
- `cluster_evaluation_kX.txt` - Clustering evaluation metrics
- `cluster_sentiment_crosstab_kX_absolute.csv` - Sentiment distribution (absolute)
- `cluster_sentiment_crosstab_kX_normalized.csv` - Sentiment distribution (normalized)
- `visualizations/` - PNG plots (cluster size, sentiment distribution, silhouette analysis)
- `cluster_profiles/` - Detailed profiles for each cluster (.txt and .xlsx)

**Cluster Profile Files** (`cluster_N_profile.xlsx`):
- Sheet1 `keywords`: Top keywords for the cluster (TF-IDF ranked)
- Sheet2 `comments`: Sample comments from the cluster

## Technical Stack
- **Python 3.11+** (3.11.12 recommended)
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (KMeans, TF-IDF, metrics)
- **Text Embeddings**: sentence-transformers (SBERT)
- **Visualization**: matplotlib
- **File I/O**: openpyxl (Excel), pickle

## Usage
Run the clustering analysis with default parameters:
```bash
cd sdr_clustering_analysis
python main.py
```

### Command Line Options
- `--k`: Specify K value for K-means (default: automatic selection)
- `--k_min`, `--k_max`: Set K range for automatic selection
- `--use_preprocessing`: Enable text preprocessing
- `--no_embedding_cache`: Disable embedding cache
- `--algorithm`: Clustering algorithm (currently only 'kmeans')

Example: Run with K=5 and preprocessing enabled:
```bash
python main.py --k 5 --use_preprocessing
```

## Maintenance Notes
- Maintain unique comment IDs in raw data for traceability
- SBERT embeddings are cached to avoid recomputation
- All output files preserve original comment indexing for consistency
- The framework can be extended to other comment analysis platforms