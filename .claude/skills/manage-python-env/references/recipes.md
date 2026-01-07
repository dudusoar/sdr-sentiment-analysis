# Environment Recipes

## Quick Start Templates

### Basic Data Science Environment
```bash
# Create project
mkdir data-science-project
cd data-science-project
uv init

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install core data science packages
uv add pandas numpy scipy scikit-learn matplotlib seaborn jupyter

# Install development tools
uv add --dev pytest black flake8 jupyterlab

# Create requirements files
uv pip freeze > requirements.txt
uv pip freeze --dev > requirements-dev.txt

# Run Jupyter
uv run jupyter lab
```

### Web API Development (FastAPI)
```bash
# Create project
mkdir fastapi-project
cd fastapi-project
uv init

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install FastAPI and dependencies
uv add fastapi[all] uvicorn[standard] pydantic[email] sqlalchemy psycopg2-binary redis

# Install development tools
uv add --dev pytest httpx black isort mypy pre-commit

# Install testing tools
uv add --dev pytest-asyncio pytest-cov pytest-xdist

# Create project structure
mkdir -p src/app/{api,models,services,database} tests
touch src/app/__init__.py src/app/main.py

# Run development server
uv run uvicorn src.app.main:app --reload
```

### Machine Learning Project
```bash
# Create project
mkdir ml-project
cd ml-project
uv init

# Create environment with specific Python version
uv venv --python 3.10  # Some ML packages work better with 3.10
source .venv/bin/activate

# Install ML frameworks
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv add tensorflow scikit-learn xgboost lightgbm catboost

# Install data processing
uv add pandas numpy scipy matplotlib seaborn plotly

# Install ML utilities
uv add scikit-optimize hyperopt optuna mlflow wandb

# Install development tools
uv add --dev pytest black flake8 jupyter notebook

# For GPU support (if available)
# uv add torch torchvision torchaudio --index-url https://download.pytytorch.org/whl/cu118
```

### CLI Tool Development
```bash
# Create project
mkdir cli-tool
cd cli-tool
uv init

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install CLI frameworks
uv add click typer rich prompt-toolkit

# Install utilities
uv add pyyaml tomlkit requests

# Install packaging tools
uv add --dev build twine

# Install testing
uv add --dev pytest pytest-mock pytest-click

# Create CLI structure
mkdir -p src/cli_tool/{commands,utils} tests
touch src/cli_tool/__init__.py src/cli_tool/cli.py

# Install in development mode
uv add -e .
```

## Platform-Specific Recipes

### Windows Development
```bash
# Windows-specific packages
uv add pywin32 pythonnet comtypes

# Windows performance
uv config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"

# Create Windows-friendly environment
uv venv --copies  # Copy Python binaries for better isolation

# Activate on Windows
.venv\Scripts\activate
```

### macOS Development
```bash
# macOS system dependencies (install first)
# brew install openssl readline sqlite3 xz zlib

# Set compilation flags for some packages
export LDFLAGS="-L$(brew --prefix openssl)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include"

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install packages that need compilation
uv add cryptography psycopg2-binary
```

### Linux Development
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev build-essential libssl-dev libffi-dev

# Create isolated environment
uv venv --isolated
source .venv/bin/activate

# Install packages
uv add cryptography psycopg2-binary
```

## Docker Integration

### Minimal Dockerfile
```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv && uv sync --frozen

# Copy application code
COPY . .

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run application
CMD ["python", "main.py"]
```

### Multi-stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder

RUN pip install uv
WORKDIR /app

COPY pyproject.toml ./
RUN uv venv && uv sync --frozen

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "main.py"]
```

### Docker Compose with uv
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - uv-cache:/root/.cache/uv
    environment:
      - UV_CACHE_DIR=/root/.cache/uv
    command: uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  uv-cache:
```

## CI/CD Recipes

### GitHub Actions
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: uv sync --frozen
          
      - name: Run tests
        run: uv run pytest --cov --cov-report=xml
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### GitLab CI
```yaml
test:
  image: python:3.11-slim
  before_script:
    - pip install uv
    - uv venv
    - uv sync --frozen
  script:
    - uv run pytest
```

### Azure Pipelines
```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
      
  - script: |
      pip install uv
      uv venv
      uv sync --frozen
      uv run pytest
    displayName: 'Test with uv'
```

## Migration Recipes

### Migrate from pip/venv
```bash
# 1. Install uv
pip install uv

# 2. Create new environment with uv
uv venv --python 3.11
source .venv/bin/activate

# 3. Install packages from existing requirements.txt
uv add -r requirements.txt

# 4. Test everything works
uv run pytest

# 5. (Optional) Convert to pyproject.toml
# Create pyproject.toml from requirements
cat > pyproject.toml << EOF
[project]
name = "project-name"
version = "0.1.0"
dependencies = [
    # Add dependencies from requirements.txt
]
EOF

# 6. Use uv sync going forward
uv sync
```

### Migrate from Poetry
```bash
# 1. Install uv
pip install uv

# 2. Create environment
uv venv --python 3.11
source .venv/bin/activate

# 3. Install from pyproject.toml
uv add -r pyproject.toml

# 4. Generate requirements.txt (optional)
uv pip freeze > requirements.txt

# 5. Update .gitignore
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore

# 6. Remove Poetry files
rm poetry.lock pyproject.toml  # Backup first!
```

### Migrate from pipenv
```bash
# 1. Install uv
pip install uv

# 2. Export Pipenv requirements
pipenv requirements > requirements.txt
pipenv requirements --dev > requirements-dev.txt

# 3. Create environment
uv venv --python 3.11
source .venv/bin/activate

# 4. Install packages
uv add -r requirements.txt
uv add --dev -r requirements-dev.txt

# 5. Remove Pipenv files
rm Pipfile Pipfile.lock
```

## Advanced Recipes

### Multiple Environment Strategy
```bash
# Create different environments for different purposes

# Development environment
uv venv .venv-dev
source .venv-dev/bin/activate
uv sync --dev

# Testing environment
uv venv .venv-test --python 3.11
source .venv-test/bin/activate
uv sync --group test

# Production environment (minimal)
uv venv .venv-prod --python 3.11 --isolated
source .venv-prod/bin/activate
uv sync --no-dev

# Switch between environments
deactivate
source .venv-dev/bin/activate  # Development
source .venv-test/bin/activate # Testing
source .venv-prod/bin/activate # Production
```

### Dependency Pinning Strategy
```bash
# requirements.in (unpinned dependencies)
requests>=2.28.0
pandas>=1.5.0
numpy>=1.24.0

# Compile to pinned requirements.txt
uv pip compile requirements.in -o requirements.txt

# Update periodically
uv pip compile requirements.in -o requirements.txt --upgrade

# For security updates only
uv pip compile requirements.in -o requirements.txt --upgrade-package security-package
```

### Local Package Development
```bash
# Setup for developing multiple related packages

# Package A (library)
cd package-a
uv venv
uv add -e .

# Package B (depends on A)
cd ../package-b
uv venv
uv add -e ../package-a  # Editable install from local path
uv add -e .

# Both packages in same environment
cd ../project
uv venv
uv add -e ../package-a
uv add -e ../package-b
uv add -e .
```

### Offline Development
```bash
# 1. On machine with internet
uv pip download -r requirements.txt -d ./packages

# 2. Copy packages directory to offline machine
# 3. On offline machine
uv venv
uv add --no-index --find-links ./packages -r requirements.txt

# 4. For new packages, repeat download step
```

### Version Matrix Testing
```bash
# Test with multiple Python versions
for version in 3.9 3.10 3.11; do
    echo "Testing Python $version"
    uv venv --python $version .venv-$version
    source .venv-$version/bin/activate
    uv sync --frozen
    uv run pytest
    deactivate
done
```

## Troubleshooting Recipes

### Fix Broken Environment
```bash
# 1. Backup current environment
cp -r .venv .venv-backup

# 2. Create fresh environment
uv venv --python 3.11 .venv-new
source .venv-new/bin/activate

# 3. Reinstall packages
uv sync

# 4. Test
uv run pytest

# 5. If works, replace old environment
deactivate
rm -rf .venv
mv .venv-new .venv
source .venv/bin/activate
```

### Resolve Dependency Conflicts
```bash
# 1. Check conflict details
uv pip check

# 2. Show dependency tree
uv pip list --tree

# 3. Try different resolution strategies
uv sync --resolution=highest
uv sync --resolution=lowest-direct

# 4. If still failing, try removing conflicting package
uv remove conflicting-package
uv add "conflicting-package<problematic-version"

# 5. Use constraints file
echo "conflicting-package<2.0.0" > constraints.txt
uv sync -c constraints.txt
```

### Optimize Installation Speed
```bash
# 1. Use faster index
uv config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"

# 2. Enable parallel downloads
uv config set download.concurrency 20

# 3. Use connection pooling
uv config set download.pool-connections true

# 4. Cache wheels
uv config set cache.wheels true

# 5. Pre-download packages for CI
uv pip download -r requirements.txt -d ./packages
uv sync --find-links ./packages
```