# Topic Modeling Analysis

## Overview
This module performs topic modeling analysis on YouTube delivery robot comments using Latent Dirichlet Allocation (LDA). It identifies latent topics in comment data, evaluates optimal topic numbers, generates topic keywords, and provides comprehensive visualizations of topic distributions.

## Directory Structure
```
topic_modeling/
├── data/
│   └── comments.xlsx                     # Input comment data with pure_text column
├── topic_model_results/                  # Main output directory
│   ├── models/                           # Saved LDA models for different topic numbers
│   │   └── lda_model_{1-20}.gensim       # LDA model files
│   ├── topic_visualization/              # Topic keyword distribution visualizations
│   │   └── topic_keywords_distribution_{5-10}.png
│   ├── topic_distribution_under_sentiment_label/  # Topic distribution by sentiment
│   │   └── label_{0-2}_texts_topic_distribution.png
│   ├── coherence_score.png               # Coherence score vs. topics plot
│   └── perplexity_score.png              # Perplexity vs. topics plot
├── topic_modeling_analysis.py            # Main analysis script
├── analyze_topic.py                      # Topic analysis and visualization utilities
├── topic_results_and_probability_distribution.ipynb  # Jupyter notebook for detailed analysis
└── __pycache__/                          # Python cache files
```

## Main Functions
1. **Data Preprocessing**: Text cleaning, contraction expansion, stopword removal, lemmatization
2. **Topic Modeling**: LDA implementation using gensim with asymmetric alpha and auto eta
3. **Model Evaluation**: Coherence and perplexity scoring to determine optimal topic numbers
4. **Visualization**: Topic keyword distributions, coherence/perplexity plots, topic-word heatmaps
5. **Sentiment Integration**: Topic distribution analysis under different sentiment labels

## Input Data Format
- **comments.xlsx**: Excel file containing comment data with a `pure_text` column
- Each row represents a YouTube comment about delivery robots
- Text preprocessing handles contractions, punctuation, stopwords, and domain-specific terms

## Processing Pipeline
1. **Data Loading**: Load comments from Excel file
2. **Text Preprocessing**:
   - Lowercasing and contraction expansion
   - Tokenization and punctuation removal
   - Stopword filtering (including domain-specific terms like "robot", "delivery", etc.)
   - Lemmatization using pattern.en
3. **Corpus Preparation**: Create dictionary and bag-of-words corpus
4. **Optimal Topic Discovery**:
   - Train LDA models for topic numbers 1-20
   - Compute coherence (c_v) and perplexity scores
   - Plot evaluation metrics to identify optimal topic count
5. **Model Training**: Train and save LDA models for all topic numbers
6. **Topic Analysis**: Extract keywords and importance weights for each topic
7. **Visualization**: Generate topic keyword distribution plots and heatmaps

## Output Files
- **Saved Models**: LDA models for topic numbers 1-20 in `topic_model_results/models/`
- **Evaluation Plots**: `coherence_score.png` and `perplexity_score.png`
- **Topic Visualizations**: `topic_keywords_distribution_{N}.png` for various topic counts
- **Sentiment Analysis**: Topic distribution plots for different sentiment labels
- **Topic Analysis Results**: Detailed keyword distributions and heatmaps

## Key Features
- **Domain-Specific Stopwords**: Custom word exclusion list for delivery robot context
- **Flexible Topic Range**: Configurable topic number range (default: 1-20)
- **Model Persistence**: All trained models saved for future analysis
- **Comprehensive Evaluation**: Coherence and perplexity metrics for model selection
- **Sentiment Integration**: Analysis of topic distributions across sentiment categories

## Usage
### Main Analysis
```bash
# Run the main topic modeling analysis
python topic_modeling_analysis.py

# Customize topic number range
python topic_modeling_analysis.py --start_topics 5 --end_topics 15
```

### Topic Analysis
```bash
# Analyze specific saved model (e.g., 10 topics)
python analyze_topic.py
```

### Notebook Analysis
Open `topic_results_and_probability_distribution.ipynb` for interactive analysis and visualization.

## Dependencies
- pandas, numpy
- nltk (stopwords, wordnet, punkt)
- gensim (LDA implementation)
- matplotlib (visualization)
- pattern.en (lemmatization)
- openpyxl (Excel file handling)

## Integration with YouTube-SC Project
This module is part of the YouTube-SC (YouTube Sentiment and Clustering Analysis) project. It works with comment data processed by sentiment classification modules and provides topic modeling insights that complement clustering analysis results. The sentiment-labeled topic distributions enable cross-analysis between topic content and emotional tone.

## Sample Topics Identified
Based on analysis of 10 topics:
1. Job security concerns
2. Food delivery applications
3. Technology of the future
4. Food security concerns
5. Human negative interactions
6. Cross intersection concerns
7. Conflict with pedestrian concerns
8. Potential usages
9. SDR security concerns
10. Support SDRs but mixed concerns