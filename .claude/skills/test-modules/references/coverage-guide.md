# Test Coverage Guide for YouTube-SC

## Introduction

Test coverage measures how much of your code is executed during testing. For the YouTube-SC project, maintaining high test coverage ensures code quality and reduces bugs.

## Installation

```bash
# Install pytest-cov
pip install pytest-cov

# Already included in YouTube-SC requirements.txt
```

## Basic Usage

### Running Tests with Coverage
```bash
# Basic coverage for entire project
pytest --cov=.

# Coverage for specific module
pytest --cov=sdr_clustering_analysis

# Coverage with HTML report
pytest --cov=. --cov-report=html

# Multiple coverage reports
pytest --cov=. --cov-report=html --cov-report=xml --cov-report=term
```

### Coverage Configuration (.coveragerc)
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
    @abstractmethod
    @property
    @staticmethod
    @classmethod

fail_under = 80
```

## Coverage Reports

### Terminal Report
```bash
pytest --cov=. --cov-report=term
```
Output shows:
- Files with coverage percentages
- Missing lines
- Summary statistics

### HTML Report
```bash
pytest --cov=. --cov-report=html
```
Generates `htmlcov/` directory with:
- Interactive file browser
- Line-by-line coverage highlighting
- Summary dashboard

### XML Report
```bash
pytest --cov=. --cov-report=xml
```
Generates `coverage.xml` for CI/CD integration.

### JSON Report
```bash
pytest --cov=. --cov-report=json
```
Generates `coverage.json` for programmatic analysis.

## Module-Specific Coverage

### Clustering Module Coverage
```bash
# Coverage for clustering module only
pytest tests/sdr_clustering_analysis/ --cov=sdr_clustering_analysis --cov-report=term

# With minimum coverage threshold
pytest --cov=sdr_clustering_analysis --cov-fail-under=85
```

### Sentiment Module Coverage
```bash
# ML sentiment coverage
pytest tests/sentiment_classification_ML/ --cov=sentiment_classification_ML

# BERT sentiment coverage
pytest tests/sentiment_classification_Bert/ --cov=sentiment_classification_Bert
```

### Topic Modeling Coverage
```bash
pytest tests/topic_modeling/ --cov=topic_modeling
```

### Text Statistics Coverage
```bash
pytest tests/text_statistics/ --cov=text_statistics
```

## Coverage Analysis

### Identifying Untested Code
```bash
# Show missing lines
pytest --cov=. --cov-report=term-missing

# Example output:
# Name                          Stmts   Miss  Cover   Missing
# -----------------------------------------------------------
# module.py                       100     20    80%   24-30, 45-50, 78-85
```

### Branch Coverage
```bash
# Measure branch coverage (if/else, loops)
pytest --cov=. --cov-branch
```

### Path Coverage
```bash
# Measure path coverage
pytest --cov=. --cov-context=test
```

## Coverage for YouTube-SC Modules

### Clustering Module Targets
```python
# sdr_clustering_analysis/
# Target: 85% coverage
# Critical paths:
# - Data loading and validation
# - Clustering algorithms
# - Result visualization
# - Error handling

# .coveragerc addition for clustering
[run]
source = sdr_clustering_analysis
omit =
    sdr_clustering_analysis/__main__.py
    sdr_clustering_analysis/legacy/*

[report]
exclude_lines =
    pragma: clustering experimental
```

### Sentiment Module Targets
```python
# sentiment_classification_ML/ and sentiment_classification_Bert/
# Target: 80% coverage
# Critical paths:
# - Text preprocessing
# - Feature extraction
# - Model training/inference
# - Prediction post-processing

# ML-specific exclusions
[report:ml_sentiment]
exclude_lines =
    pragma: ml experimental
    # Large data processing (test with samples)
    if len(data) > 10000

# BERT-specific exclusions
[report:bert_sentiment]
exclude_lines =
    pragma: bert download
    # HuggingFace model loading (mocked in tests)
    from_pretrained
```

## Coverage Exclusions

### Justified Exclusions
```python
# 1. Debug code
if DEBUG:  # pragma: no cover
    print("Debug info")

# 2. Platform-specific code
if sys.platform == "win32":  # pragma: no cover
    windows_specific()

# 3. Experimental features
def experimental_feature():  # pragma: experimental
    """Not ready for production."""

# 4. Type checking blocks
if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional

# 5. Abstract methods
@abstractmethod
def must_implement(self):  # pragma: no cover
    """Child classes must implement."""

# 6. Property getters/setters
@property
def computed_value(self):  # pragma: no cover
    return self._value * 2
```

### YouTube-SC Specific Exclusions
```python
# Data loading (test with sample data)
def load_large_dataset(path):  # pragma: no cover large data
    """Loads GBs of data - test with samples."""
    return pd.read_csv(path)

# Visualization (test logic, not display)
def show_plot():  # pragma: no cover visualization
    plt.show()  # Don't test display

# External API calls
def call_external_api():  # pragma: no cover external
    response = requests.get(...)  # Mock in tests
```

## Coverage in CI/CD

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests with coverage
      run: |
        pytest --cov=. --cov-report=xml --cov-report=term

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Coverage Thresholds
```yaml
# In CI script
- name: Check coverage threshold
  run: |
    pytest --cov=. --cov-fail-under=80
  # Fails if coverage < 80%
```

## Improving Coverage

### Identifying Gaps
```bash
# Generate detailed missing lines report
pytest --cov=. --cov-report=term-missing > coverage_gaps.txt

# Analyze by module
pytest --cov=sdr_clustering_analysis --cov-report=term-missing
```

### Writing Missing Tests

#### Example: Untested Error Handling
```python
# Original code (untested)
def process_data(data):
    if not data:
        raise ValueError("No data provided")  # Untested
    return len(data)

# Test to add
def test_process_data_empty():
    """Test error handling for empty data."""
    with pytest.raises(ValueError, match="No data provided"):
        process_data([])
```

#### Example: Untested Edge Cases
```python
# Original code (untested edge case)
def calculate_ratio(a, b):
    if b == 0:
        return float('inf')  # Untested
    return a / b

# Test to add
def test_calculate_ratio_zero_denominator():
    """Test edge case with zero denominator."""
    assert calculate_ratio(10, 0) == float('inf')
```

## Coverage Best Practices

### 1. Focus on Critical Paths
- Prioritize testing core algorithms
- Ensure error handling is tested
- Test integration points between modules

### 2. Use Coverage Intelligently
- Don't chase 100% blindly
- Exclude genuinely untestable code
- Focus on risk areas

### 3. Maintain Coverage History
```bash
# Track coverage over time
pytest --cov=. --cov-report=json --cov-report=term
# Save JSON for trend analysis
```

### 4. Module-Specific Standards
```python
# Clustering: High coverage (85%) - algorithms critical
# Sentiment: Good coverage (80%) - ML models complex
# Topic Modeling: Good coverage (80%) - statistical methods
# Text Statistics: Moderate coverage (75%) - simple calculations
```

## Coverage Tools Integration

### VS Code Integration
```json
// .vscode/settings.json
{
    "python.testing.pytestArgs": [
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-coverage
        name: Run tests with coverage
        entry: pytest --cov=. --cov-fail-under=80
        language: system
        pass_filenames: false
        always_run: true
```

## Monitoring Coverage Trends

### Coverage History Script
```python
# scripts/coverage_tracker.py
import json
import datetime
import matplotlib.pyplot as plt

def track_coverage():
    # Run tests and get coverage
    # Parse coverage.json
    # Store in history file
    # Generate trend graph
    pass
```

### Coverage Dashboard
```html
<!-- Generate simple HTML dashboard -->
<html>
<body>
<h1>YouTube-SC Coverage Dashboard</h1>
<ul>
<li>Clustering: 85%</li>
<li>ML Sentiment: 80%</li>
<li>BERT Sentiment: 75%</li>
<li>Topic Modeling: 82%</li>
<li>Text Statistics: 78%</li>
<li>Overall: 80%</li>
</ul>
</body>
</html>
```

## Common Coverage Issues

### False Positives
```python
# Code that runs but coverage doesn't detect
def generator_function():
    yield 1  # May not be detected
    yield 2

# Solution: Test generator consumption
def test_generator():
    gen = generator_function()
    assert list(gen) == [1, 2]
```

### Async Code Coverage
```python
# Async functions need special handling
async def async_function():
    return await some_operation()

# Test with pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### Property Coverage
```python
# Properties may not be detected
class MyClass:
    @property
    def value(self):
        return self._value

# Test property access
def test_property():
    obj = MyClass()
    obj._value = 10
    assert obj.value == 10
```

## YouTube-SC Coverage Goals

### Quarterly Targets
| Module | Q1 Target | Q2 Target | Q3 Target | Notes |
|--------|-----------|-----------|-----------|-------|
| Clustering | 80% | 85% | 90% | Core algorithms |
| ML Sentiment | 75% | 80% | 85% | Training pipeline |
| BERT Sentiment | 70% | 75% | 80% | Model inference |
| Topic Modeling | 75% | 80% | 85% | LDA implementation |
| Text Statistics | 70% | 75% | 80% | Analysis functions |
| **Overall** | **75%** | **80%** | **85%** | **Project goal** |

### Action Plan
1. **Current Assessment**: Run coverage to establish baseline
2. **Priority Areas**: Identify low-coverage critical paths
3. **Test Creation**: Write tests for missing coverage
4. **Continuous Monitoring**: Integrate into CI/CD
5. **Improvement Cycles**: Quarterly coverage reviews