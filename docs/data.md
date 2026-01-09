# Data Description

This document describes the dataset used in the YouTube-SC project, including data sources, processing steps, structure, and ethical considerations.

## Data Source
The dataset consists of YouTube comments collected from videos related to sidewalk delivery robots (SDRs). The comments were collected from publicly available YouTube videos discussing delivery robot technologies, deployments, and public reactions.

## Dataset Files
The project uses two main data sources located in the `data/` directory:

### 1. Main Comment Dataset
- **File**: `data/combined_comments.xlsx`
- **Format**: Excel spreadsheet with multiple sheets
- **Content**: Raw and processed YouTube comments with sentiment labels
- **Size**: Contains thousands of comments across multiple videos

### 2. Video Statistics
- **Directory**: `data/Video_statistics/`
- **Content**: Additional metadata and statistics about the source videos
- **Purpose**: Supplementary information for analysis context

## Data Processing Pipeline

### Step 1: Raw Comment Collection
- Comments extracted from YouTube videos using the YouTube Data API
- Initial collection includes comment text, timestamps, and basic metadata

### Step 2: Cleaning and Filtering
- Irrelevant comments, advertisements, and non-textual entries were removed
- Duplicate comments and spam content filtered out
- Language filtering to focus on English comments for BERT analysis

### Step 3: Text Preprocessing
The following preprocessing steps are applied to comment text:
1. **Case Folding**: Convert all text to lowercase
2. **Tokenization**: Split text into individual tokens/words
3. **Stopword Removal**: Remove common English stopwords (using NLTK)
4. **Lemmatization**: Reduce words to their base/dictionary form
5. **Special Character Handling**: Preserve negation words and selected punctuation for sentiment analysis

### Step 4: Manual Annotation
Each comment is manually annotated into one of three sentiment categories:
- **0**: Negative sentiment
- **1**: Positive sentiment
- **2**: Neutral sentiment

Annotation was performed by trained annotators following clear guidelines to ensure consistency.

## Data Structure

### combined_comments.xlsx Structure
The main dataset contains the following columns (exact column names may vary):

| Column Name | Description | Data Type |
|------------|-------------|-----------|
| `comment_text` | Original comment text | String |
| `processed_text` | Preprocessed comment text | String |
| `sentiment_label` | Manual sentiment annotation (0, 1, 2) | Integer |


### Additional Data Files
- **Training/Validation/Test Splits**: Located in module-specific directories for machine learning experiments
- **Feature Files**: Pre-computed TF-IDF vectors, Word2Vec embeddings, and other features
- **Model Outputs**: Prediction results and analysis outputs in module `results/` directories

## Data Flow
The dataset follows this processing flow through the analysis pipeline:

```
Raw Comments (data/combined_comments.xlsx)
        ↓
    [Processing & Annotation]
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

## Privacy and Ethical Considerations

To protect user privacy and comply with platform policies:

### Anonymization
- All video identifiers and user-related metadata have been removed or anonymized
- User names, profile information, and direct identifiers are excluded
- Only anonymized comment text and derived labels/features are released

### Ethical Use
- The dataset is intended strictly for research purposes
- Comments are from publicly available videos but treated with respect for user privacy
- Analysis focuses on aggregate patterns rather than individual comments
- No attempt is made to identify or contact individual users

### Compliance
- Collection and use comply with YouTube Terms of Service
- Research follows institutional review board (IRB) guidelines for human subjects research
- Data sharing follows academic norms for reproducible research while protecting privacy

## Usage Notes

### For Researchers
1. The dataset supports reproducible research in sentiment analysis, topic modeling, and social media analysis
2. Preprocessing scripts are provided to ensure consistent processing across studies
3. Annotated sentiment labels enable supervised learning experiments

### For Developers
1. The data structure is designed for easy integration with Python data analysis libraries (pandas, numpy)
2. Module-specific data loaders are provided in each analysis module
3. Configuration files allow easy adjustment of data paths and parameters
