# Development Recommendations

This document provides recommendations and best practices for developing and extending the YouTube-SC project. These guidelines help maintain code quality, performance, and consistency across the codebase.

## Development Environment Setup

### Recommended Tools

#### Code Editor/IDE
- **VS Code**: With Python extension, Pylance, and Black Formatter
- **PyCharm**: Professional edition for full Python development features
- **Jupyter Lab**: For exploratory analysis and prototyping

#### Essential Extensions
```bash
# VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension eamodio.gitlens
code --install-extension yzhang.markdown-all-in-one

# PyCharm plugins
# Python, Markdown, GitToolBox, Rainbow Brackets
```

#### Development Dependencies
```bash
# Install development tools
uv add --dev black flake8 pytest pytest-cov mypy pre-commit jupyter

# Optional: additional tools
uv add --dev ipython matplotlib seaborn  # For interactive exploration
uv add --dev bandit safety  # Security scanning
uv add --dev pytest-xdist  # Parallel test execution
```

## Code Quality

### Code Formatting
Use **Black** for consistent code formatting:
```bash
# Format all Python files
black .

# Format specific module
black sentiment_classification_ML/

# Check formatting without applying
black --check .
```

**Black Configuration** (optional `pyproject.toml`):
```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

### Code Linting
Use **Flake8** with plugins for comprehensive linting:
```bash
# Run flake8
flake8 .

# With specific configuration
flake8 --max-line-length=88 --extend-ignore=E203,W503
```

**Flake8 Configuration** (`.flake8`):
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist,.venv,.vscode
per-file-ignores =
    __init__.py: F401
```

### Type Checking
Use **mypy** for static type checking:
```bash
# Run mypy on all files
mypy .

# Run on specific module
mypy sentiment_classification_ML/
```

**Type Hint Examples**:
```python
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from sklearn.svm import SVC

def preprocess_text(
    text: str,
    remove_stopwords: bool = True
) -> str:
    """
    Preprocess text with optional stopword removal.

    Parameters:
    -----------
    text : str
        Input text to preprocess
    remove_stopwords : bool, default=True
        Whether to remove stopwords

    Returns:
    --------
    str
        Preprocessed text
    """
    # Implementation
    return processed_text

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "svm"
) -> Union[SVC, Any]:
    """
    Train machine learning model.
    """
    # Implementation
    return model
```

## Testing Strategy

### Test Organization
```
tests/
├── unit/                    # Unit tests
│   ├── test_data_loader.py
│   ├── test_feature_extractor.py
│   └── test_models.py
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   └── test_module_integration.py
├── e2e/                    # End-to-end tests
│   └── test_full_pipeline.py
└── conftest.py             # Shared fixtures
```

### Writing Effective Tests

#### Unit Tests
```python
import pytest
import pandas as pd
from my_module.data_loader import load_comments

def test_load_comments_basic():
    """Test basic comment loading functionality."""
    df = load_comments("data/sample.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "comment_text" in df.columns

def test_load_comments_empty_file():
    """Test loading empty file."""
    with pytest.raises(ValueError, match="No data loaded"):
        load_comments("data/empty.csv")

@pytest.fixture
def sample_data():
    """Fixture providing sample data for multiple tests."""
    return pd.DataFrame({
        "comment_text": ["Great video!", "Not helpful"],
        "sentiment": [1, 0]
    })

def test_process_with_fixture(sample_data):
    """Test using fixture data."""
    processed = process_comments(sample_data)
    assert len(processed) == len(sample_data)
```

#### Integration Tests
```python
def test_sentiment_pipeline_integration():
    """Test complete sentiment analysis pipeline."""
    # Load data
    data = load_comments("data/test.csv")

    # Preprocess
    processed = preprocess_comments(data)

    # Extract features
    features = extract_features(processed)

    # Train model
    model = train_model(features, data["sentiment"])

    # Make predictions
    predictions = model.predict(features)

    # Verify results
    assert len(predictions) == len(data)
    assert predictions.dtype == int
```

#### Performance Tests
```python
import pytest
import time

def test_training_performance():
    """Test that training completes within time limit."""
    start_time = time.time()

    # Training code
    train_model(large_dataset)

    elapsed = time.time() - start_time
    assert elapsed < 300, "Training took too long (300s limit)"

@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of large dataset (marked as slow)."""
    # This test will be skipped by default
    process_large_dataset()
```

### Test Execution
```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run fast tests only (skip slow ones)
python -m pytest -m "not slow"

# Run in parallel
python -m pytest -n auto

# Generate test report
python -m pytest --html=report.html --self-contained-html
```

## Performance Optimization

### Memory Optimization

#### Use Sparse Matrices
```python
from scipy.sparse import csr_matrix

# For TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(texts)  # Returns sparse matrix

# Convert to dense only when necessary
X_dense = X_sparse.toarray()  # Only if algorithm requires dense
```

#### Process Data in Chunks
```python
def process_large_file(filepath, chunk_size=1000):
    """Process large file in chunks to avoid memory issues."""
    results = []

    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        processed = process_chunk(chunk)
        results.append(processed)

    return pd.concat(results, ignore_index=True)

# Usage
large_data = process_large_file("data/large_comments.csv", chunk_size=5000)
```

#### Use Generators
```python
def comment_generator(filepath):
    """Yield comments one at a time."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield process_line(line)

# Usage
for comment in comment_generator("data/comments.txt"):
    process_comment(comment)
```

### Speed Optimization

#### Vectorized Operations
```python
# Avoid loops - use vectorized operations
import numpy as np
import pandas as pd

# Slow
result = []
for text in df['comment_text']:
    result.append(len(text.split()))

# Fast
df['word_count'] = df['comment_text'].str.split().str.len()

# Even faster with NumPy
word_counts = np.array([len(text.split()) for text in df['comment_text'].values])
```

#### Parallel Processing
```python
from multiprocessing import Pool
import functools

def process_comment(comment, param1, param2):
    """Process single comment."""
    return processed_comment

def process_comments_parallel(comments, n_jobs=4):
    """Process comments in parallel."""
    with Pool(n_jobs) as pool:
        # Partial function to fix additional parameters
        process_func = functools.partial(process_comment, param1=value1, param2=value2)
        results = pool.map(process_func, comments)
    return results

# Usage with joblib (simpler)
from joblib import Parallel, delayed
results = Parallel(n_jobs=4)(delayed(process_comment)(c) for c in comments)
```

#### Caching Results
```python
import joblib
from functools import lru_cache

# Disk caching for expensive computations
@joblib.Memory(location='./cachedir').cache
def expensive_computation(data):
    """Expensive computation cached to disk."""
    return result

# Memory caching for frequently called functions
@lru_cache(maxsize=128)
def load_stopwords(language='english'):
    """Load stopwords with caching."""
    from nltk.corpus import stopwords
    return set(stopwords.words(language))
```

### Accuracy Optimization

#### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid search for optimal parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(), param_grid, refit=True, verbose=3, cv=5
)
grid_search.fit(X_train, y_train)

# Randomized search for larger parameter spaces
random_search = RandomizedSearchCV(
    SVC(), param_distributions, n_iter=100, cv=5, verbose=3
)
```

#### Ensemble Methods
```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('svm', SVC(probability=True)),
        ('rf', RandomForestClassifier()),
        ('nb', MultinomialNB())
    ],
    voting='soft'
)

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('svm', SVC()),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

## Configuration Management

### Environment-Specific Configurations
```python
# config.py
import os

class Config:
    # Development configuration
    DEVELOPMENT = {
        'DATA_PATH': 'data/sample.csv',
        'MODEL_SAVE_PATH': 'models/dev/',
        'LOG_LEVEL': 'DEBUG',
        'CACHE_SIZE': 1000
    }

    # Production configuration
    PRODUCTION = {
        'DATA_PATH': 'data/full_dataset.csv',
        'MODEL_SAVE_PATH': 'models/prod/',
        'LOG_LEVEL': 'INFO',
        'CACHE_SIZE': 10000
    }

    @classmethod
    def get_config(cls):
        env = os.getenv('ENVIRONMENT', 'DEVELOPMENT')
        return getattr(cls, env.upper())

# Usage
config = Config.get_config()
data_path = config['DATA_PATH']
```

### Configuration Validation
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ModelConfig(BaseModel):
    """Model configuration with validation."""
    model_type: str = Field(..., regex='^(svm|rf|nb|dt)$')
    n_estimators: Optional[int] = Field(None, ge=10, le=1000)
    learning_rate: Optional[float] = Field(None, gt=0, le=1)

    @validator('n_estimators')
    def validate_estimators(cls, v, values):
        if values.get('model_type') == 'rf' and v is None:
            raise ValueError('n_estimators required for Random Forest')
        return v

# Usage
config = ModelConfig(model_type='rf', n_estimators=100)
```

## Documentation Best Practices

### Code Documentation
```python
def calculate_sentiment_score(
    text: str,
    model: SentimentModel,
    threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Calculate sentiment score for given text.

    Parameters:
    -----------
    text : str
        Input text to analyze
    model : SentimentModel
        Trained sentiment model
    threshold : float, default=0.5
        Confidence threshold for classification

    Returns:
    --------
    Tuple[int, float]
        Sentiment label (0, 1, 2) and confidence score

    Raises:
    -------
    ValueError
        If text is empty or model is not trained

    Examples:
    ---------
    >>> model = load_model('sentiment.model')
    >>> label, confidence = calculate_sentiment_score("Great video!", model)
    >>> print(f"Label: {label}, Confidence: {confidence:.2f}")
    Label: 1, Confidence: 0.87
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")

    # Implementation
    return label, confidence
```

### API Documentation
Use **Sphinx** or **MkDocs** for API documentation:
```bash
# Install documentation tools
uv add --dev sphinx sphinx-rtd-theme myst-parser

# Generate documentation
sphinx-apidoc -o docs/source .
sphinx-build -b html docs/source docs/build
```

## Version Control Practices

### Commit Messages
Follow conventional commits format:
```
feat: add new clustering visualization
fix: resolve memory leak in data loader
docs: update installation instructions
test: add unit tests for sentiment analysis
refactor: simplify configuration management
chore: update dependencies
```

### Branch Strategy
```bash
# Feature branches
git checkout -b feat/new-analysis-method
git checkout -b fix/issue-123

# Release branches
git checkout -b release/v1.2.0

# Hotfix branches
git checkout -b hotfix/critical-bug
```

### Git Hooks
Use pre-commit hooks for automated checks:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
```

## Security Considerations

### Input Validation
```python
import html

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # HTML escape
    text = html.escape(text)

    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+='  # Remove event handlers
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()
```

### Secure Configuration
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Never hardcode sensitive information
api_key = os.getenv('API_KEY')
database_url = os.getenv('DATABASE_URL')

# Use secrets module for cryptographic operations
import secrets
secret_key = secrets.token_hex(32)
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: CI Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -r requirements.txt
          uv pip install -r requirements-dev.txt
      - name: Run tests
        run: python -m pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Run linters
        run: |
          pip install black flake8 mypy
          black --check .
          flake8 .
          mypy .
```

## Monitoring and Logging

### Structured Logging
```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Structured logging
def log_training_start(model_name, dataset_size):
    logger.info(
        "Training started",
        extra={
            "model": model_name,
            "dataset_size": dataset_size,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

### Performance Monitoring
```python
import time
from functools import wraps

def timeit(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        logger.info(
            f"{func.__name__} executed in {end_time - start_time:.4f} seconds",
            extra={
                "function": func.__name__,
                "execution_time": end_time - start_time
            }
        )

        return result
    return wrapper

# Usage
@timeit
def train_model(data):
    # Training code
    pass
```

Following these development recommendations will help maintain high code quality, performance, and maintainability throughout the YouTube-SC project lifecycle.