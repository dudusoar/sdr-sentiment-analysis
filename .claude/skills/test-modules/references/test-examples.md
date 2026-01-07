# Test Examples for YouTube-SC

This document provides concrete test examples for the YouTube-SC project modules.

## Test Data Management

### Sample Data Fixtures
```python
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_comments():
    """Sample YouTube comments for testing."""
    return [
        "This delivery robot is amazing!",
        "I hate when it gets stuck",
        "Good service overall",
        "Could be faster",
        "Love the efficiency!"
    ]

@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        'comment': ['Great!', 'Bad', 'Average', 'Excellent'],
        'sentiment': [1, -1, 0, 1],
        'length': [5, 3, 7, 8]
    })

@pytest.fixture
def sample_numpy_data():
    """Sample numpy array for clustering."""
    return np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [5.0, 6.0],
        [5.1, 6.1],
        [10.0, 11.0]
    ])
```

## Clustering Module Tests

### Data Loading Tests
```python
import pytest
from sdr_clustering_analysis.data_loader import load_comments_from_excel

def test_load_comments_success(tmp_path):
    """Test successful comment loading."""
    # Create test Excel file
    test_file = tmp_path / "test_comments.xlsx"
    df = pd.DataFrame({'comment': ['test1', 'test2']})
    df.to_excel(test_file, index=False)

    # Load and verify
    comments = load_comments_from_excel(str(test_file))
    assert len(comments) == 2
    assert 'test1' in comments
    assert 'test2' in comments

def test_load_comments_empty_file(tmp_path):
    """Test loading empty Excel file."""
    test_file = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(test_file, index=False)

    with pytest.raises(ValueError, match="No comments found"):
        load_comments_from_excel(str(test_file))

def test_load_comments_missing_file():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_comments_from_excel("nonexistent.xlsx")
```

### Clustering Algorithm Tests
```python
import pytest
from sdr_clustering_analysis.clustering import kmeans_cluster, calculate_silhouette

def test_kmeans_clustering(sample_numpy_data):
    """Test K-means clustering."""
    n_clusters = 3
    labels, centroids = kmeans_cluster(sample_numpy_data, n_clusters)

    assert len(labels) == len(sample_numpy_data)
    assert len(centroids) == n_clusters
    assert set(labels) == set(range(n_clusters))

def test_kmeans_invalid_k(sample_numpy_data):
    """Test K-means with invalid k value."""
    with pytest.raises(ValueError, match="k must be positive"):
        kmeans_cluster(sample_numpy_data, k=0)

    with pytest.raises(ValueError, match="k cannot exceed"):
        kmeans_cluster(sample_numpy_data, k=100)

def test_silhouette_score_range(sample_numpy_data):
    """Test silhouette score is in valid range."""
    labels = [0, 0, 1, 1, 2]
    score = calculate_silhouette(sample_numpy_data, labels)

    assert -1 <= score <= 1

def test_silhouette_single_cluster(sample_numpy_data):
    """Test silhouette with single cluster."""
    labels = [0] * len(sample_numpy_data)
    score = calculate_silhouette(sample_numpy_data, labels)
    assert score == 0  # Defined as 0 for single cluster
```

### Visualization Tests
```python
import pytest
from matplotlib import pyplot as plt
from sdr_clustering_analysis.visualization import plot_clusters

def test_plot_clusters_creation(sample_numpy_data, mocker):
    """Test cluster plot creation."""
    # Mock plt.show to avoid displaying plot
    mock_show = mocker.patch('matplotlib.pyplot.show')
    labels = [0, 0, 1, 1, 2]

    # Create plot
    fig = plot_clusters(sample_numpy_data, labels)

    assert fig is not None
    assert len(fig.axes) > 0
    mock_show.assert_not_called()  # Should not show in tests

def test_plot_clusters_save(tmp_path, sample_numpy_data):
    """Test saving cluster plot."""
    output_file = tmp_path / "clusters.png"
    labels = [0, 0, 1, 1, 2]

    fig = plot_clusters(sample_numpy_data, labels, save_path=str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
```

## Sentiment Classification Tests

### Text Preprocessing Tests
```python
import pytest
from sentiment_classification_ML.preprocessing import clean_text, tokenize_text

def test_clean_text():
    """Test text cleaning function."""
    # Test lowercase conversion
    assert clean_text("HELLO World") == "hello world"

    # Test punctuation removal
    assert clean_text("Hello, world!") == "hello world"

    # Test number removal
    assert clean_text("test 123") == "test"

    # Test multiple spaces removal
    assert clean_text("hello   world") == "hello world"

def test_clean_text_empty():
    """Test cleaning empty text."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""

def test_tokenize_text():
    """Test text tokenization."""
    tokens = tokenize_text("hello world test")
    assert tokens == ["hello", "world", "test"]

def test_tokenize_with_stopwords():
    """Test tokenization with stopword removal."""
    tokens = tokenize_text("this is a test", remove_stopwords=True)
    assert "test" in tokens
    assert "this" not in tokens  # Stopword removed
```

### Feature Extraction Tests
```python
import pytest
from sentiment_classification_ML.features import extract_tfidf, extract_word_counts

def test_extract_tfidf(sample_comments):
    """Test TF-IDF feature extraction."""
    features = extract_tfidf(sample_comments)

    assert features.shape[0] == len(sample_comments)
    assert features.shape[1] > 0  # Should have some features

def test_extract_tfidf_custom_vocab(sample_comments):
    """Test TF-IDF with custom vocabulary."""
    vocabulary = ["robot", "service", "delivery"]
    features = extract_tfidf(sample_comments, vocabulary=vocabulary)

    assert features.shape[1] == len(vocabulary)

def test_extract_word_counts(sample_comments):
    """Test word count feature extraction."""
    features = extract_word_counts(sample_comments)

    assert features.shape[0] == len(sample_comments)
    assert "robot" in features.columns

def test_extract_word_counts_top_n(sample_comments):
    """Test word counts with top N words."""
    features = extract_word_counts(sample_comments, top_n=5)

    assert features.shape[1] == 5
```

### Model Training Tests
```python
import pytest
from sklearn.model_selection import train_test_split
from sentiment_classification_ML.training import train_sentiment_model

def test_train_sentiment_model(sample_comments):
    """Test sentiment model training."""
    # Create dummy labels
    labels = [1, -1, 1, 0, 1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sample_comments, labels, test_size=0.2, random_state=42
    )

    # Train model
    model, accuracy = train_sentiment_model(X_train, y_train, X_test, y_test)

    assert model is not None
    assert 0 <= accuracy <= 1

def test_train_model_empty_data():
    """Test training with empty data."""
    with pytest.raises(ValueError, match="No training data"):
        train_sentiment_model([], [], [], [])
```

## BERT Sentiment Tests

### Model Loading Tests
```python
import pytest
from sentiment_classification_Bert.model import load_bert_model

def test_load_bert_model(mocker):
    """Test BERT model loading."""
    # Mock transformers to avoid downloading
    mock_auto = mocker.patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    mock_tokenizer = mocker.patch('transformers.AutoTokenizer.from_pretrained')

    model, tokenizer = load_bert_model()

    mock_auto.assert_called_once()
    mock_tokenizer.assert_called_once()
    assert model is not None
    assert tokenizer is not None

def test_load_bert_model_custom_path(mocker):
    """Test loading BERT model from custom path."""
    mock_auto = mocker.patch('transformers.AutoModelForSequenceClassification.from_pretrained')

    model = load_bert_model(model_path="custom/model")

    mock_auto.assert_called_with("custom/model")
```

### Tokenization Tests
```python
import pytest
from sentiment_classification_Bert.tokenization import tokenize_comments

def test_tokenize_comments(sample_comments, mocker):
    """Test comment tokenization."""
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.return_value = {
        'input_ids': [[1, 2, 3]],
        'attention_mask': [[1, 1, 1]]
    }

    tokens = tokenize_comments(sample_comments, mock_tokenizer, max_length=128)

    assert 'input_ids' in tokens
    assert 'attention_mask' in tokens
    assert len(tokens['input_ids']) == len(sample_comments)

def test_tokenize_empty_comments():
    """Test tokenizing empty comment list."""
    with pytest.raises(ValueError, match="No comments to tokenize"):
        tokenize_comments([], None)
```

### Inference Tests
```python
import pytest
import torch
from sentiment_classification_Bert.inference import predict_sentiment

def test_predict_sentiment(mocker):
    """Test sentiment prediction."""
    # Mock model
    mock_model = mocker.Mock()
    mock_model.return_value.logits = torch.tensor([[1.0, -1.0, 0.0]])

    # Mock tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }

    predictions = predict_sentiment(["test comment"], mock_model, mock_tokenizer)

    assert len(predictions) == 1
    assert predictions[0] in ['positive', 'negative', 'neutral']

def test_predict_sentiment_batch(mocker):
    """Test batch prediction."""
    mock_model = mocker.Mock()
    mock_model.return_value.logits = torch.tensor([
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0]
    ])

    comments = ["comment 1", "comment 2"]
    predictions = predict_sentiment(comments, mocker.Mock(), mock_model)

    assert len(predictions) == 2
```

## Topic Modeling Tests

### LDA Model Tests
```python
import pytest
from topic_modeling.lda_model import train_lda_model, get_topic_keywords

def test_train_lda_model(sample_comments):
    """Test LDA model training."""
    num_topics = 3
    model, coherence = train_lda_model(sample_comments, num_topics=num_topics)

    assert model is not None
    assert 0 <= coherence <= 1
    assert len(model.show_topics()) == num_topics

def test_train_lda_invalid_topics(sample_comments):
    """Test LDA with invalid topic count."""
    with pytest.raises(ValueError, match="Number of topics must be positive"):
        train_lda_model(sample_comments, num_topics=0)

def test_get_topic_keywords(mocker):
    """Test topic keyword extraction."""
    mock_model = mocker.Mock()
    mock_model.show_topic.return_value = [("robot", 0.5), ("delivery", 0.3)]

    keywords = get_topic_keywords(mock_model, topic_id=0, num_words=5)

    assert len(keywords) == 2
    assert keywords[0][0] == "robot"
    assert keywords[0][1] == 0.5
```

### Topic Coherence Tests
```python
import pytest
from topic_modeling.coherence import calculate_coherence

def test_calculate_coherence(sample_comments):
    """Test topic coherence calculation."""
    # Mock model with topics
    class MockModel:
        def show_topics(self):
            return [
                (0, "0.5*robot + 0.3*delivery + 0.2*service"),
                (1, "0.6*slow + 0.4*fast")
            ]

    model = MockModel()
    coherence = calculate_coherence(model, sample_comments)

    assert 0 <= coherence <= 1

def test_calculate_coherence_no_topics():
    """Test coherence with no topics."""
    class MockModel:
        def show_topics(self):
            return []

    with pytest.raises(ValueError, match="No topics found"):
        calculate_coherence(MockModel(), [])
```

## Text Statistics Tests

### Text Analysis Tests
```python
import pytest
from text_statistics.analysis import calculate_word_frequencies, analyze_sentiment_distribution

def test_calculate_word_frequencies(sample_comments):
    """Test word frequency calculation."""
    frequencies = calculate_word_frequencies(sample_comments, top_n=5)

    assert len(frequencies) <= 5
    assert all(isinstance(word, str) for word in frequencies.keys())
    assert all(isinstance(count, int) for count in frequencies.values())

def test_calculate_word_frequencies_empty():
    """Test word frequencies with empty comments."""
    frequencies = calculate_word_frequencies([])

    assert len(frequencies) == 0

def test_analyze_sentiment_distribution():
    """Test sentiment distribution analysis."""
    sentiments = [1, -1, 0, 1, 1, -1, 0]
    distribution = analyze_sentiment_distribution(sentiments)

    assert 'positive' in distribution
    assert 'negative' in distribution
    assert 'neutral' in distribution
    assert sum(distribution.values()) == len(sentiments)
```

### Statistical Calculation Tests
```python
import pytest
from text_statistics.stats import calculate_text_stats, get_longest_words

def test_calculate_text_stats(sample_comments):
    """Test text statistics calculation."""
    stats = calculate_text_stats(sample_comments)

    assert 'total_comments' in stats
    assert 'avg_length' in stats
    assert 'max_length' in stats
    assert 'min_length' in stats

    assert stats['total_comments'] == len(sample_comments)
    assert stats['avg_length'] > 0

def test_calculate_text_stats_empty():
    """Test statistics with empty comments."""
    stats = calculate_text_stats([])

    assert stats['total_comments'] == 0
    assert stats['avg_length'] == 0

def test_get_longest_words(sample_comments):
    """Test finding longest words."""
    longest_words = get_longest_words(sample_comments, n=3)

    assert len(longest_words) == 3
    assert all(isinstance(word, str) for word in longest_words)
```

## Integration Tests

### Cross-Module Integration Tests
```python
import pytest
from sdr_clustering_analysis.clustering import kmeans_cluster
from sentiment_classification_ML.inference import predict_sentiment_batch

def test_clustering_with_sentiment(sample_comments, sample_numpy_data):
    """Test integration of clustering and sentiment analysis."""
    # Cluster comments
    labels, _ = kmeans_cluster(sample_numpy_data, k=2)

    # Analyze sentiment for each cluster
    cluster_sentiments = {}
    for cluster_id in set(labels):
        cluster_comments = [sample_comments[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        sentiments = predict_sentiment_batch(cluster_comments)
        cluster_sentiments[cluster_id] = sum(sentiments) / len(sentiments)

    assert len(cluster_sentiments) == 2
    assert all(-1 <= s <= 1 for s in cluster_sentiments.values())
```

### End-to-End Pipeline Tests
```python
import pytest
from youtube_sc.pipeline import run_analysis_pipeline

def test_analysis_pipeline(tmp_path, mocker):
    """Test complete analysis pipeline."""
    # Mock individual module functions
    mocker.patch('sdr_clustering_analysis.load_data', return_value=(["comment"], [[1,2]]))
    mocker.patch('sdr_clustering_analysis.cluster', return_value=([0, 1], [[1,2],[3,4]]))
    mocker.patch('sentiment_classification_ML.analyze', return_value=[1, -1])
    mocker.patch('topic_modeling.extract_topics', return_value=[("topic1", 0.5)])

    # Run pipeline
    output_file = tmp_path / "results.json"
    results = run_analysis_pipeline("dummy_input.xlsx", str(output_file))

    assert output_file.exists()
    assert 'clusters' in results
    assert 'sentiments' in results
    assert 'topics' in results
```

## Performance Tests

### Timing Tests
```python
import pytest
import time
from sdr_clustering_analysis.clustering import kmeans_cluster

@pytest.mark.slow
def test_clustering_performance_large_data():
    """Test clustering performance with large dataset."""
    # Generate large dataset
    large_data = np.random.randn(1000, 10)

    start_time = time.time()
    labels, centroids = kmeans_cluster(large_data, k=5)
    elapsed_time = time.time() - start_time

    assert elapsed_time < 10.0  # Should complete within 10 seconds
    assert len(labels) == 1000
```

### Memory Usage Tests
```python
import pytest
import psutil
import os
from sentiment_classification_Bert.model import load_bert_model

@pytest.mark.slow
def test_bert_memory_usage(mocker):
    """Test BERT model memory usage."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Mock actual model loading to avoid GPU memory
    mocker.patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    mocker.patch('transformers.AutoTokenizer.from_pretrained')

    load_bert_model()

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
```

## Best Practices Demonstrated

1. **Isolated Tests**: Each test focuses on specific functionality
2. **Mocking**: External dependencies are mocked
3. **Edge Cases**: Tests include empty input, invalid parameters
4. **Fixtures**: Shared test data through fixtures
5. **Performance**: Tests for timing and memory usage
6. **Integration**: Tests for cross-module functionality
7. **Error Handling**: Tests for exception conditions