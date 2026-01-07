# YouTube-SC Testing Examples

This directory contains testing examples and configurations specific to the YouTube-SC project.

## Test Setup

### 1. Project Structure for Testing
```
Youtube-SC/
├── tests/                          # Test directory
│   ├── conftest.py                 # Shared fixtures
│   ├── sdr_clustering_analysis/    # Clustering tests
│   ├── sentiment_classification_ML/ # ML sentiment tests
│   ├── sentiment_classification_Bert/ # BERT sentiment tests
│   ├── topic_modeling/             # Topic modeling tests
│   └── text_statistics/            # Text statistics tests
├── Code/                           # Source code
└── .claude/                        # Configuration
```

### 2. Configuration Files

#### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: slow running tests
    integration: integration tests
    clustering: clustering module tests
    sentiment: sentiment module tests
    topic: topic modeling tests
    stats: text statistics tests
    bert: BERT model tests
```

#### .coveragerc
```ini
[run]
source = .
omit =
    .venv/*
    __pycache__/*
    tests/*
    */test_*.py
    */tests/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    raise AssertionError
    raise NotImplementedError
    if TYPE_CHECKING:

fail_under = 80
```

### 3. Shared Fixtures (conftest.py)
```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
def sample_clustering_data():
    """Sample data for clustering tests."""
    return np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [5.0, 6.0],
        [5.1, 6.1],
        [10.0, 11.0]
    ])
```

## Module Test Examples

### Clustering Module Tests
```python
# tests/sdr_clustering_analysis/test_clustering.py
import pytest
import numpy as np
from sdr_clustering_analysis.clustering import kmeans_cluster

@pytest.mark.clustering
class TestClustering:
    def test_kmeans_basic(self, sample_clustering_data):
        """Test basic K-means clustering."""
        n_clusters = 3
        labels, centroids = kmeans_cluster(sample_clustering_data, n_clusters)

        assert len(labels) == len(sample_clustering_data)
        assert len(centroids) == n_clusters
        assert set(labels) == set(range(n_clusters))

    def test_kmeans_invalid_k(self, sample_clustering_data):
        """Test K-means with invalid k value."""
        with pytest.raises(ValueError, match="k must be positive"):
            kmeans_cluster(sample_clustering_data, k=0)

    @pytest.mark.slow
    def test_kmeans_large_dataset(self):
        """Test K-means with large dataset (slow test)."""
        large_data = np.random.randn(1000, 10)
        labels, _ = kmeans_cluster(large_data, k=5)
        assert len(set(labels)) == 5
```

### ML Sentiment Module Tests
```python
# tests/sentiment_classification_ML/test_preprocessing.py
import pytest
from sentiment_classification_ML.preprocessing import clean_text, tokenize_text

@pytest.mark.sentiment
class TestTextPreprocessing:
    def test_clean_text(self):
        """Test text cleaning function."""
        assert clean_text("HELLO, World!") == "hello world"
        assert clean_text("Test 123 numbers") == "test numbers"
        assert clean_text("   multiple   spaces   ") == "multiple spaces"

    def test_tokenize_text(self):
        """Test text tokenization."""
        tokens = tokenize_text("hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_with_stopwords(self):
        """Test tokenization with stopword removal."""
        tokens = tokenize_text("this is a test", remove_stopwords=True)
        assert "test" in tokens
        assert "this" not in tokens
```

### BERT Sentiment Module Tests
```python
# tests/sentiment_classification_Bert/test_inference.py
import pytest
from unittest.mock import Mock, patch
from sentiment_classification_Bert.inference import predict_sentiment

@pytest.mark.sentiment
@pytest.mark.bert
class TestBERTInference:
    def test_predict_sentiment(self, mocker):
        """Test sentiment prediction with BERT."""
        # Mock model and tokenizer
        mock_model = mocker.Mock()
        mock_model.return_value.logits = [[1.0, -1.0, 0.0]]

        mock_tokenizer = mocker.Mock()
        mock_tokenizer.return_value = {
            'input_ids': [[1, 2, 3]],
            'attention_mask': [[1, 1, 1]]
        }

        predictions = predict_sentiment(["test comment"], mock_model, mock_tokenizer)

        assert len(predictions) == 1
        assert predictions[0] in ['positive', 'negative', 'neutral']

    @pytest.mark.slow
    def test_predict_sentiment_batch(self):
        """Test batch sentiment prediction (slow due to model loading)."""
        # This test would load actual model
        pass  # Implement with actual model for integration testing
```

### Topic Modeling Tests
```python
# tests/topic_modeling/test_lda.py
import pytest
from topic_modeling.lda_model import train_lda_model

@pytest.mark.topic
class TestLDAModel:
    def test_train_lda_model(self, sample_comments):
        """Test LDA model training."""
        num_topics = 3
        model, coherence = train_lda_model(sample_comments, num_topics=num_topics)

        assert model is not None
        assert 0 <= coherence <= 1
        assert len(model.show_topics()) == num_topics

    def test_train_lda_invalid_topics(self, sample_comments):
        """Test LDA with invalid topic count."""
        with pytest.raises(ValueError, match="Number of topics must be positive"):
            train_lda_model(sample_comments, num_topics=0)
```

### Text Statistics Tests
```python
# tests/text_statistics/test_analysis.py
import pytest
from text_statistics.analysis import calculate_word_frequencies

@pytest.mark.stats
class TestTextAnalysis:
    def test_word_frequencies(self, sample_comments):
        """Test word frequency calculation."""
        frequencies = calculate_word_frequencies(sample_comments, top_n=5)

        assert len(frequencies) <= 5
        assert all(isinstance(word, str) for word in frequencies.keys())
        assert all(isinstance(count, int) for count in frequencies.values())

    def test_word_frequencies_empty(self):
        """Test word frequencies with empty comments."""
        frequencies = calculate_word_frequencies([])
        assert len(frequencies) == 0
```

## Integration Tests

### Cross-Module Integration
```python
# tests/integration/test_pipeline.py
import pytest
from sdr_clustering_analysis.clustering import kmeans_cluster
from sentiment_classification_ML.inference import predict_sentiment_batch

@pytest.mark.integration
@pytest.mark.slow
class TestIntegration:
    def test_clustering_with_sentiment(self, sample_comments, sample_clustering_data):
        """Test integration of clustering and sentiment analysis."""
        # Cluster comments
        labels, _ = kmeans_cluster(sample_clustering_data, k=2)

        # Analyze sentiment for each cluster
        cluster_sentiments = {}
        for cluster_id in set(labels):
            cluster_comments = [
                sample_comments[i]
                for i, lbl in enumerate(labels)
                if lbl == cluster_id
            ]
            sentiments = predict_sentiment_batch(cluster_comments)
            cluster_sentiments[cluster_id] = sum(sentiments) / len(sentiments)

        assert len(cluster_sentiments) == 2
        assert all(-1 <= s <= 1 for s in cluster_sentiments.values())
```

## Running Tests

### Basic Test Commands
```bash
# Run all tests
pytest

# Run tests for specific module
pytest tests/sdr_clustering_analysis/

# Run tests with specific marker
pytest -m clustering
pytest -m sentiment
pytest -m "not slow"  # Exclude slow tests

# Run with coverage
pytest --cov=. --cov-report=html
```

### CI/CD Integration
```bash
# GitHub Actions example
pytest --cov=. --cov-report=xml --cov-fail-under=80

# Generate coverage badge
pytest --cov=. --cov-report=json
# Process JSON to generate badge
```

## Test Data Management

### Sample Data Files
Create sample data files for testing:
```python
# tests/test_data/sample_comments.json
[
    "Positive comment about delivery",
    "Negative experience with robot",
    "Neutral feedback on service"
]

# tests/test_data/sample_dataset.csv
comment,sentiment,length
"Great service",1,12
"Poor delivery",-1,13
"Average experience",0,16
```

### Mock External Dependencies
```python
# Mocking external API calls
@pytest.fixture
def mock_requests(mocker):
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {'data': 'test'}
    return mock_get

# Mocking file operations
@pytest.fixture
def mock_file_operations(mocker):
    mock_open = mocker.patch('builtins.open', mocker.mock_open(read_data='test content'))
    return mock_open
```

## Best Practices for YouTube-SC Testing

### 1. Module Isolation
- Keep tests specific to each module
- Mock dependencies between modules
- Test module interfaces thoroughly

### 2. Test Data
- Use small, representative sample data
- Create fixtures for common test data
- Avoid using production data in tests

### 3. Performance
- Mark slow tests with `@pytest.mark.slow`
- Use `pytest -m "not slow"` for fast feedback
- Mock heavy computations in unit tests

### 4. Coverage Targets
- Clustering: 85% (core algorithms)
- Sentiment: 80% (ML models)
- Topic Modeling: 80% (statistical methods)
- Text Statistics: 75% (analysis functions)

### 5. Continuous Integration
- Run tests on every commit
- Enforce coverage thresholds
- Generate test reports

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure PYTHONPATH includes project root
2. **Missing dependencies**: Install required packages from requirements.txt
3. **Slow tests**: Use markers to exclude slow tests during development
4. **Flaky tests**: Identify and fix non-deterministic tests
5. **Coverage gaps**: Write tests for uncovered critical paths

### Debugging Tests
```bash
# Run specific test with debug output
pytest tests/test_file.py::test_function -vvs

# Run with pdb on failure
pytest --pdb

# Show captured output
pytest -s
```

## Next Steps

1. **Setup test infrastructure**: Create `tests/` directory structure
2. **Write core tests**: Start with critical path tests for each module
3. **Establish coverage baseline**: Measure current coverage
4. **Improve coverage**: Write tests for uncovered code
5. **CI/CD integration**: Set up automated testing pipeline
6. **Monitor and maintain**: Regularly review and update tests