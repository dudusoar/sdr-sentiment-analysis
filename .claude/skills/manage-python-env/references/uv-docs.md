# uv Command Reference

## Installation

### Install uv
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip (not recommended but works)
pip install uv

# Using Homebrew (macOS)
brew install uv
```

### Update uv
```bash
# Self-update
uv self update

# Check version
uv --version
```

## Virtual Environments

### Create Environment
```bash
# Basic environment
uv venv

# With specific Python version
uv venv --python 3.11
uv venv --python 3.10.8

# Named environment
uv venv myenv

# Isolated environment (copies Python binaries)
uv venv --isolated

# Copy system packages
uv venv --system-site-packages

# Quiet mode
uv venv -q
```

### Manage Environments
```bash
# List environments
uv venv list

# Remove environment
uv venv remove myenv
uv venv remove .venv

# Activate environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Deactivate environment
deactivate
```

### Python Version Management
```bash
# List available Python versions
uv python list

# Download specific Python version
uv python download 3.11.0

# Set default Python version
uv python pin 3.11
```

## Package Management

### Install Packages
```bash
# Install single package
uv add requests
uv add "requests>=2.28.0"

# Install with extras
uv add "requests[security,socks]"
uv add "pandas[performance]"

# Install development packages
uv add --dev pytest
uv add --dev black flake8 mypy

# Install from file
uv add -r requirements.txt
uv add -c constraints.txt

# Install editable (development mode)
uv add -e .
uv add -e ./mypackage

# Install from URL or local file
uv add ./local-package.whl
uv add https://example.com/package.tar.gz
```

### Remove Packages
```bash
# Remove package
uv remove requests

# Remove development package
uv remove --dev pytest

# Remove multiple packages
uv remove requests pandas numpy
```

### Update Packages
```bash
# Update all packages
uv update

# Update specific package
uv update requests

# Update to specific version
uv update "requests==2.29.0"

# Update with constraints
uv update --pre           # Include pre-releases
uv update --upgrade       # Upgrade all packages
uv update --outdated      # Show outdated packages first
```

### List Packages
```bash
# List installed packages
uv pip list

# List in tree format
uv pip list --tree

# List outdated packages
uv pip list --outdated

# Show package details
uv pip show requests
```

## Dependency Management

### Sync Environment
```bash
# Sync with pyproject.toml or requirements.txt
uv sync

# Sync with frozen requirements (exact versions)
uv sync --frozen

# Sync development dependencies
uv sync --dev

# Sync with specific groups
uv sync --group test --group docs

# Sync without installing
uv sync --dry-run
```

### Freeze Dependencies
```bash
# Generate requirements.txt
uv pip freeze > requirements.txt

# Freeze with hashes
uv pip freeze --all --hashes > requirements.txt

# Freeze development dependencies
uv pip freeze --dev > requirements-dev.txt
```

### Compile Requirements
```bash
# Compile from requirements.in
uv pip compile requirements.in -o requirements.txt

# Compile with platform constraints
uv pip compile pyproject.toml --platform linux --platform macos --platform windows

# Compile with Python version
uv pip compile pyproject.toml --python-version 3.11

# Compile with resolution strategy
uv pip compile pyproject.toml --resolution=highest
uv pip compile pyproject.toml --resolution=lowest-direct
```

## Project Management

### Initialize Project
```bash
# Create new project
uv init myproject
cd myproject

# Initialize with specific template
uv init --template minimal
uv init --template package
uv init --template app
```

### Build and Publish
```bash
# Build package
uv build

# Build specific formats
uv build --wheel
uv build --sdist

# Publish to PyPI
uv publish

# Publish with specific repository
uv publish --repository testpypi
```

### Run Scripts
```bash
# Run Python script
uv run script.py
uv run -m module.name

# Run with environment
uv run --with requests --with pandas script.py

# Run development server
uv run --dev uvicorn main:app --reload
```

## Configuration

### uv Configuration
```bash
# Set configuration
uv config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
uv config set install.no-binary ":all:"

# Get configuration
uv config get global.index-url

# List all configurations
uv config list

# Edit configuration file
uv config edit
```

### Cache Management
```bash
# Show cache information
uv cache info

# Clear cache
uv cache clean

# Clear specific cache
uv cache clean http
uv cache clean wheels

# Prune cache (remove unused)
uv cache prune
```

## Advanced Features

### Lock Files
```bash
# Generate lock file
uv lock

# Generate lock file with groups
uv lock --group dev --group test

# Update lock file
uv lock --update

# Check lock file consistency
uv lock --check
```

### Platform-Specific Dependencies
```bash
# Install platform-specific packages
uv add "pywin32; sys_platform == 'win32'"
uv add "cryptography; python_version >= '3.6'"

# Compile with platform markers
uv pip compile requirements.in --platform manylinux2014_x86_64
```

### Dependency Groups
```toml
# In pyproject.toml
[project.optional-dependencies]
dev = ["pytest", "black"]
test = ["pytest-cov", "tox"]
docs = ["sphinx", "sphinx-rtd-theme"]
```

```bash
# Install specific groups
uv sync --group dev --group test

# List available groups
uv pip list --groups
```

### Hash Verification
```bash
# Add package with hash
uv add "requests==2.28.2" \
    --hash sha256:abc123... \
    --hash sha256:def456...

# Install with hash verification
uv sync --require-hashes
```

## Integration

### With pip
```bash
# Use uv as drop-in replacement for pip
uv pip install package
uv pip uninstall package
uv pip freeze

# All pip commands work with uv pip
```

### With pip-tools
```bash
# Replace pip-compile
uv pip compile requirements.in

# Replace pip-sync
uv sync
```

### With Poetry
```bash
# Convert from Poetry
uv pip compile pyproject.toml

# Install Poetry projects
uv add -r pyproject.toml
```

### With pdm
```bash
# Convert from PDM
uv pip compile pyproject.toml

# Install PDM projects
uv sync
```

## Performance Tips

### Parallel Downloads
```bash
# uv automatically uses parallel downloads
# Configure number of parallel downloads
uv config set download.concurrency 20
```

### Connection Pooling
```bash
# Reuse connections (enabled by default)
uv config set download.pool-connections true
uv config set download.pool-maxsize 10
```

### Cache Optimization
```bash
# Use local cache (enabled by default)
# Set cache location
uv config set cache.dir /path/to/cache

# Increase cache size
uv config set cache.max-size "10GB"
```

## Troubleshooting Commands

### Debug Installation
```bash
# Verbose output
uv add package -v
uv add package -vv  # More verbose

# Dry run
uv add package --dry-run

# Show what would be installed
uv add package --print
```

### Diagnostic Commands
```bash
# Check environment
uv doctor

# Show dependency graph
uv pip graph

# Check for conflicts
uv pip check

# Show environment information
uv pip debug
```

### Repair Commands
```bash
# Reinstall package
uv add --reinstall package

# Force reinstall
uv add --force-reinstall package

# Fix broken dependencies
uv sync --reinstall
```

## Common Workflows

### Development Workflow
```bash
# 1. Clone project
git clone project
cd project

# 2. Create environment
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv sync --dev

# 4. Make changes and test
uv run pytest

# 5. Update dependencies
uv update
uv pip freeze > requirements.txt

# 6. Clean up
uv cache prune
```

### CI/CD Workflow
```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create environment
uv venv --python 3.11

# 3. Install dependencies
uv sync --frozen

# 4. Run tests
uv run pytest --cov

# 5. Build package
uv build
```

### Migration Workflow
```bash
# 1. Install uv
pip install uv

# 2. Create new environment
uv venv --python 3.11

# 3. Install from existing requirements
uv add -r requirements.txt

# 4. Test everything works
uv run pytest

# 5. Switch to pyproject.toml (optional)
# Create pyproject.toml from requirements
# uv init --from-requirements
```