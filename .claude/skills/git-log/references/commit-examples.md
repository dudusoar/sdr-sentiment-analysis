# Commit Message Examples

## Conventional Commit Format

The conventional commit format provides a standardized way to write commit messages:
```
<type>(<scope>): <description>

<body>

<footer>
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks, build process, etc.

## Good Examples

### Feature Commits
```bash
# Simple feature
feat: add user authentication system

# Feature with scope
feat(auth): implement OAuth 2.0 login

# Feature with detailed body
feat(api): add REST endpoints for user management

Adds the following endpoints:
- POST /api/users - Create new user
- GET /api/users - List all users
- GET /api/users/:id - Get user by ID
- PUT /api/users/:id - Update user
- DELETE /api/users/:id - Delete user

Resolves issue #123
```

### Bug Fix Commits
```bash
# Simple bug fix
fix: resolve memory leak in data processor

# Bug fix with scope
fix(memory): prevent buffer overflow in image processing

# Bug fix with issue reference
fix: correct null pointer exception in user validation

Fixes #456 where user validation would crash when
encountering null input values.
```

### Documentation Commits
```bash
# Documentation update
docs: update API documentation with examples

# README update
docs(readme): add installation instructions

# Code comments
docs: add docstrings to all public functions

Add comprehensive docstrings following Google style
guide for better code documentation.
```

### Refactoring Commits
```bash
# Simple refactor
refactor: extract validation logic into separate module

# Performance refactor
refactor(performance): optimize database query in user search

# Code cleanup
refactor: clean up unused imports and variables

Remove all unused imports and variables identified
by the linter to improve code quality.
```

### Test Commits
```bash
# Add tests
test: add unit tests for authentication module

# Fix tests
test: fix flaky integration tests

# Test coverage
test: increase test coverage for data processing module

Adds comprehensive test coverage for:
- Data validation functions
- Transformation pipelines
- Error handling scenarios
```

### Chore Commits
```bash
# Dependency update
chore: update dependencies to latest versions

# Build process
chore: update webpack configuration for production

# CI/CD
chore: add GitHub Actions workflow for automated testing
```

## YouTube-SC Project Examples

### Clustering Analysis
```bash
feat(clustering): implement K-means clustering algorithm

Adds K-means clustering with elbow method for
optimal cluster determination. Includes visualization
of clustering results and silhouette analysis.

Related to task #12 in project board.
```

```bash
fix(clustering): correct centroid initialization bug

Fixes issue where random centroid initialization
could lead to suboptimal clustering results.
Now uses K-means++ initialization for better
convergence.
```

### Sentiment Classification
```bash
feat(sentiment): add BERT model for sentiment analysis

Implements BERT-based sentiment classification
using pre-trained model from Hugging Face.
Includes fine-tuning on YouTube comment dataset.
```

```bash
fix(sentiment): resolve memory issue with large datasets

Optimizes memory usage when processing large
comment datasets by implementing batch processing
and garbage collection.
```

### Topic Modeling
```bash
feat(topic): implement LDA topic modeling

Adds Latent Dirichlet Allocation for topic
modeling of YouTube comments. Includes coherence
score calculation and topic visualization.
```

### Text Statistics
```bash
feat(stats): add text frequency analysis

Implements word frequency analysis, TF-IDF
calculation, and visualization of most common
terms in comment datasets.
```

## Body Writing Guidelines

### Good Body Examples
```
feat: add export functionality to results module

Adds CSV and JSON export options for analysis results:
- CSV export with customizable delimiter
- JSON export with pretty printing option
- Progress indicator during export
- Error handling for file system issues

User can now export results for external analysis
and reporting purposes.
```

```
fix: resolve encoding issue with Chinese comments

Fixes Unicode encoding problem when processing
comments containing Chinese characters:
- Specify UTF-8 encoding for all file operations
- Add encoding detection for input files
- Update text processing functions to handle Unicode

Resolves issue #78 where Chinese characters would
appear as garbled text in output.
```

### Footer Examples
```
Closes #123
Fixes #456
Related to #789

BREAKING CHANGE: API response format changed from
JSON array to object with metadata.
```

## Bad Examples to Avoid

### Too Vague
```bash
# Bad
update code
fix stuff
add things

# Good
fix: resolve null pointer in user authentication
feat: add data validation for user input
```

### Too Long Subject
```bash
# Bad
fix: resolve the issue with the data processing module that was causing memory leaks when handling large datasets with multiple nested objects

# Good
fix(memory): prevent leak in data processing with large datasets
```

### Mixing Multiple Changes
```bash
# Bad
feat: add new API and fix bugs

# Good (separate commits)
feat: add new REST API endpoints
fix: resolve validation bug in user input
```

### Imperative vs Descriptive
```bash
# Bad (imperative in body)
Add new feature for user management

# Good (descriptive in body)
Adds user management feature including create,
read, update, and delete operations.
```

## Special Cases

### Breaking Changes
```bash
feat!: change API response format

BREAKING CHANGE: The API response format has been
changed from a flat array to a nested object
structure. Migration guide available in docs.
```

### Revert Commits
```bash
revert: "feat: add experimental caching layer"

This reverts commit abc123def456 due to instability
issues in production environment.

Will re-implement with proper testing in future.
```

### Merge Commits
```bash
Merge branch 'feature/new-analysis' into main

Adds new sentiment analysis module with improved
accuracy and performance metrics.
```

## Commit Message Template

Use this template for consistent commit messages:

```bash
<type>(<scope>): <short description>

<detailed description>

<footer with issue references>
```

Example usage:
```bash
git commit -m "feat(clustering): add silhouette analysis

Adds silhouette score calculation for cluster
quality assessment. Includes visualization of
silhouette scores per sample.

Related to issue #45 in project board."
```