# pytest Guide for YouTube-SC

## Introduction

pytest is the recommended testing framework for the YouTube-SC project. It provides a simple, scalable way to write tests with minimal boilerplate.

## Installation

```bash
# Basic installation
pip install pytest

# With common plugins
pip install pytest pytest-cov pytest-mock pytest-xdist

# For YouTube-SC project (already in requirements.txt)
pip install pytest pytest-cov
```

## Basic Test Structure

### Simple Test File
```python
# test_basic.py
def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"
```

### Test Class
```python
# test_class.py
class TestCalculator:
    def test_add(self):
        assert 1 + 2 == 3

    def test_multiply(self):
        assert 2 * 3 == 6
```

## Running Tests

### Basic Commands
```bash
# Run all tests in current directory
pytest

# Run specific test file
pytest test_module.py

# Run specific test function
pytest test_module.py::test_function_name

# Run tests in specific directory
pytest tests/

# Run with verbose output
pytest -v

# Run with detailed traceback
pytest --tb=long

# Run and stop on first failure
pytest -x
```

### Test Discovery
pytest discovers tests using the following rules:
- Files named `test_*.py` or `*_test.py`
- Functions named `test_*`
- Classes named `Test*` with methods named `test_*`

## Fixtures

### Basic Fixture
```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15
```

### Fixture with Setup/Teardown
```python
import pytest
import tempfile
import os

@pytest.fixture
def temp_file():
    """Create temporary file for testing."""
    # Setup
    temp = tempfile.NamedTemporaryFile(delete=False, mode='w')
    temp.write("test content")
    temp.close()

    yield temp.name  # Provide to test

    # Teardown
    os.unlink(temp.name)

def test_file_content(temp_file):
    with open(temp_file, 'r') as f:
        content = f.read()
    assert content == "test content"
```

### Fixture Scopes
```python
@pytest.fixture(scope="function")   # Default: new for each test
def func_scope():
    return {}

@pytest.fixture(scope="class")      # Once per test class
def class_scope():
    return {}

@pytest.fixture(scope="module")     # Once per module
def module_scope():
    return {}

@pytest.fixture(scope="session")    # Once per test session
def session_scope():
    return {}
```

## Parametrization

### Basic Parametrization
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected
```

### Multiple Parameters
```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (4, 5, 9),
    (10, 20, 30),
])
def test_addition(a, b, expected):
    assert a + b == expected
```

### Combining Fixtures and Parameters
```python
import pytest

@pytest.fixture
def multiplier():
    return 2

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_with_fixture(input, expected, multiplier):
    assert input * multiplier == expected
```

## Mocking

### Using pytest-mock
```python
import pytest

def test_with_mock(mocker):
    # Mock a function
    mock_func = mocker.patch('module.function_name')
    mock_func.return_value = 'mocked'

    # Call function under test
    result = function_under_test()

    # Verify
    assert result == 'expected'
    mock_func.assert_called_once()
```

### Mock Examples
```python
import pytest
from unittest.mock import Mock

def test_mock_examples(mocker):
    # Mock with return value
    mock_download = mocker.patch('requests.get')
    mock_download.return_value.status_code = 200
    mock_download.return_value.json.return_value = {'data': 'test'}

    # Mock with side effect
    mock_process = mocker.patch('module.process_data')
    mock_process.side_effect = [1, 2, 3]

    # Mock property
    mock_obj = Mock()
    mock_obj.property_name = 'value'

    # Verify calls
    mock_download.assert_called_once_with('http://example.com')
```

## Marks and Filtering

### Built-in Marks
```python
import pytest

@pytest.mark.skip(reason="Not implemented yet")
def test_skipped():
    assert False

@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8+")
def test_skip_condition():
    assert True

@pytest.mark.xfail(reason="Known issue")
def test_expected_fail():
    assert False

@pytest.mark.slow
def test_slow_operation():
    # Long-running test
    time.sleep(5)
    assert True
```

### Custom Marks
```python
# In pytest.ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: integration tests
    clustering: tests for clustering module
    sentiment: tests for sentiment module
```

```python
# In test files
@pytest.mark.clustering
def test_clustering_algorithm():
    assert True

@pytest.mark.sentiment
def test_sentiment_analysis():
    assert True
```

### Running Marked Tests
```bash
# Run only clustering tests
pytest -m clustering

# Run all except slow tests
pytest -m "not slow"

# Run integration tests only
pytest -m integration
```

## Configuration

### pytest.ini Configuration
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
```

### Conftest.py
```python
# tests/conftest.py
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project-wide fixtures
@pytest.fixture(scope="session")
def project_data():
    """Load project data once for all tests."""
    # Load and return test data
    return load_test_data()
```

## Test Organization for YouTube-SC

### Recommended Structure
```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── sdr_clustering_analysis/       # Clustering module tests
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_clustering.py
│   └── test_visualization.py
├── sentiment_classification_ML/   # ML sentiment tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_evaluation.py
├── sentiment_classification_Bert/ # BERT sentiment tests
│   ├── __init__.py
│   ├── test_bert_model.py
│   ├── test_tokenization.py
│   └── test_inference.py
├── topic_modeling/                # Topic modeling tests
│   ├── __init__.py
│   ├── test_lda.py
│   ├── test_topics.py
│   └── test_visualization.py
└── text_statistics/               # Text statistics tests
    ├── __init__.py
    ├── test_analysis.py
    ├── test_stats.py
    └── test_export.py
```

### Module-Specific Test Patterns

#### Clustering Tests
```python
@pytest.mark.clustering
class TestClustering:
    @pytest.fixture
    def sample_data(self):
        return load_clustering_test_data()

    def test_kmeans(self, sample_data):
        result = kmeans_cluster(sample_data, k=3)
        assert len(result.labels_) == len(sample_data)

    def test_silhouette_score(self, sample_data):
        score = calculate_silhouette(sample_data)
        assert -1 <= score <= 1
```

#### Sentiment Tests
```python
@pytest.mark.sentiment
class TestSentiment:
    @pytest.fixture
    def sample_texts(self):
        return ["I love this product", "This is terrible", "It's okay"]

    def test_sentiment_scores(self, sample_texts):
        scores = analyze_sentiment(sample_texts)
        assert len(scores) == len(sample_texts)
        assert all(-1 <= s <= 1 for s in scores)
```

## Advanced Features

### Temporary Directory
```python
import pytest

def test_with_temp_dir(tmp_path):
    # tmp_path is a Path object
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    assert test_file.read_text() == "content"
    assert test_file.exists()
```

### Capturing Output
```python
def test_capture_stdout(capsys):
    print("Hello, World!")
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
```

### Exception Testing
```python
import pytest

def test_exception():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("Error message")

    assert str(exc_info.value) == "Error message"
    assert exc_info.type == ValueError
```

### Approximate Equality
```python
def test_approximate():
    result = 0.1 + 0.2
    assert result == pytest.approx(0.3)

    # With tolerance
    assert 1.0 == pytest.approx(1.001, abs=0.01)
```

## Running Tests in Parallel

### Using pytest-xdist
```bash
# Run tests on 4 CPUs
pytest -n 4

# Auto-detect number of CPUs
pytest -n auto

# Run with specific distribution
pytest --dist=loadscope
```

## Test Reports

### Generating Reports
```bash
# JUnit XML report (for CI)
pytest --junitxml=report.xml

# HTML report
pytest --html=report.html

# Plain text report
pytest --tb=line
```

### Custom Report Hooks
```python
# In conftest.py
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary to test output."""
    if exitstatus == 0:
        terminalreporter.write_sep("=", "All tests passed!")
    else:
        terminalreporter.write_sep("=", f"Tests failed: {exitstatus}")
```

## Best Practices for YouTube-SC

1. **Module Isolation**: Keep tests specific to each module
2. **Data Management**: Use fixtures for test data, avoid real data files
3. **Mock External Dependencies**: Mock API calls, file I/O, etc.
4. **Fast Tests**: Keep tests fast for quick feedback
5. **Clear Test Names**: Use descriptive test function names
6. **Test Coverage**: Aim for high coverage of core logic
7. **Continuous Integration**: Run tests automatically on changes
8. **Documentation**: Document test purpose and assumptions