# Migration Guide

## Migrating from pip/venv

### Step-by-Step Migration

#### 1. Assess Current Setup
```bash
# Check current environment
python --version
pip --version
pip list

# Check requirements files
ls requirements*.txt

# Check virtual environment location
echo $VIRTUAL_ENV
which python
```

#### 2. Install uv
```bash
# Install uv
pip install uv

# Or use install script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

#### 3. Create New Environment with uv
```bash
# Deactivate current environment if active
deactivate

# Remove old virtual environment (optional)
rm -rf venv .venv env

# Create new environment with uv
uv venv --python 3.11

# Activate
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

#### 4. Install Dependencies
```bash
# If you have requirements.txt
uv add -r requirements.txt

# If you have requirements-dev.txt
uv add --dev -r requirements-dev.txt

# If you have setup.py or setup.cfg
# Extract requirements manually first
```

#### 5. Test Migration
```bash
# Verify packages installed
uv pip list

# Run your application/tests
uv run python -c "import your_package; print('Import successful')"
uv run pytest

# Check for any issues
uv pip check
```

#### 6. Update Project Files
```bash
# Update .gitignore
cat >> .gitignore << EOF

# uv virtual environment
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
EOF

# Create pyproject.toml (optional but recommended)
cat > pyproject.toml << EOF
[project]
name = "your-project-name"
version = "0.1.0"
dependencies = [
    # Add your dependencies here
]
EOF

# Generate requirements.txt from installed packages
uv pip freeze > requirements.txt
```

#### 7. Update Documentation
Update README.md and development documentation:
```markdown
## Development Setup

### Using uv (Recommended)

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

### Using pip/venv (Legacy)

[Keep old instructions for reference]
```

### Common Migration Issues

#### 1. Package Version Differences
uv may resolve different versions than pip. To match pip's versions:

```bash
# Get exact versions from pip
pip freeze > pip-freeze.txt

# Install exact versions with uv
uv add -r pip-freeze.txt
```

#### 2. Editable Installs
```bash
# pip: pip install -e .
# uv:
uv add -e .
```

#### 3. Development Dependencies
```bash
# pip: pip install -e .[dev]
# uv:
uv add -e . --dev
```

#### 4. Constraint Files
```bash
# pip: pip install -c constraints.txt
# uv:
uv add -c constraints.txt
```

## Migrating from Poetry

### Why Migrate from Poetry to uv?
- **Speed**: uv is significantly faster
- **Simplicity**: Single tool vs Poetry + venv
- **Compatibility**: Better pip compatibility

### Migration Steps

#### 1. Backup Poetry Configuration
```bash
# Backup pyproject.toml and poetry.lock
cp pyproject.toml pyproject.toml.poetry-backup
cp poetry.lock poetry.lock.backup
```

#### 2. Install uv
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Convert pyproject.toml
Poetry's pyproject.toml is mostly compatible. Key differences:

**Before (Poetry)**:
```toml
[tool.poetry]
name = "project"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.8"
requests = "^2.28.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"
```

**After (uv/standard)**:
```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = [
    "requests>=2.28.0,<3.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0,<8.0.0",
]
```

#### 4. Create Environment and Install
```bash
# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv add -r pyproject.toml
```

#### 5. Generate Requirements Files (Optional)
```bash
# For compatibility with other tools
uv pip freeze > requirements.txt
uv pip freeze --dev > requirements-dev.txt
```

#### 6. Update CI/CD and Documentation
Replace `poetry install` with `uv sync` in:
- GitHub Actions
- Dockerfiles
- README.md
- Makefile

### Poetry to uv Command Mapping

| Poetry Command | uv Command | Notes |
|----------------|------------|-------|
| `poetry install` | `uv sync` | |
| `poetry add package` | `uv add package` | |
| `poetry add --dev package` | `uv add --dev package` | |
| `poetry remove package` | `uv remove package` | |
| `poetry update` | `uv update` | |
| `poetry run python` | `uv run python` | |
| `poetry shell` | `source .venv/bin/activate` | |
| `poetry export` | `uv pip freeze` | |

## Migrating from pipenv

### Why Migrate from pipenv to uv?
- **Performance**: uv is much faster
- **Reliability**: Better dependency resolution
- **Modern**: Actively maintained

### Migration Steps

#### 1. Export pipenv Requirements
```bash
# Ensure pipenv environment is active
pipenv shell

# Export requirements
pipenv requirements > requirements.txt
pipenv requirements --dev > requirements-dev.txt

# Exit pipenv shell
exit
```

#### 2. Install uv
```bash
# Install uv
pip install uv
# Or use install script
```

#### 3. Create New Environment
```bash
# Remove old pipenv environment
pipenv --rm

# Create uv environment
uv venv
source .venv/bin/activate
```

#### 4. Install Dependencies
```bash
# Install production dependencies
uv add -r requirements.txt

# Install development dependencies
uv add --dev -r requirements-dev.txt
```

#### 5. Remove pipenv Files
```bash
# Remove pipenv-specific files
rm Pipfile Pipfile.lock

# Update .gitignore
# Remove pipenv patterns, add uv patterns
```

### pipenv to uv Command Mapping

| pipenv Command | uv Command | Notes |
|----------------|------------|-------|
| `pipenv install` | `uv sync` | |
| `pipenv install package` | `uv add package` | |
| `pipenv install --dev package` | `uv add --dev package` | |
| `pipenv uninstall package` | `uv remove package` | |
| `pipenv update` | `uv update` | |
| `pipenv run python` | `uv run python` | |
| `pipenv shell` | `source .venv/bin/activate` | |
| `pipenv lock` | `uv pip freeze` | |
| `pipenv lock -r` | `uv pip freeze > requirements.txt` | |

## Migrating from conda

### Why Migrate from conda to uv?
- **Focus**: Python-only vs multi-language
- **Speed**: uv is faster for Python packages
- **Compatibility**: Better PyPI compatibility

### Migration Steps

#### 1. Export conda Environment
```bash
# Export conda environment
conda env export --from-history > environment.yml

# Or get pip packages from conda
conda list --export > conda-packages.txt
pip freeze > pip-packages.txt
```

#### 2. Analyze Dependencies
Conda includes system dependencies and non-Python packages. Identify Python-only packages:

```bash
# Look for packages available on PyPI
# Common conda-only packages: mkl, intel-openmp, cudatoolkit
# These may need alternative installation methods
```

#### 3. Install uv
```bash
# Install uv
pip install uv
```

#### 4. Create Python Environment
```bash
# Deactivate conda
conda deactivate

# Create uv environment
uv venv --python 3.11
source .venv/bin/activate
```

#### 5. Install Python Packages
```bash
# Install from exported pip packages
uv add -r pip-packages.txt

# For conda-only packages, find PyPI alternatives
# Example: mkl -> use numpy from PyPI with OpenBLAS
```

#### 6. Handle System Dependencies
For packages requiring system libraries:

```bash
# Linux: Use system package manager
sudo apt-get install libblas-dev liblapack-dev

# macOS: Use Homebrew
brew install openblas

# Windows: Use pre-built wheels or conda-forge
# Consider keeping some packages in conda
```

### Hybrid Approach
For projects needing both conda and pip:

```bash
# 1. Create minimal conda environment
conda create -n project python=3.11
conda activate project

# 2. Install system dependencies via conda
conda install mkl intel-openmp

# 3. Use uv for Python packages
pip install uv
uv add -r requirements.txt
```

## Migrating from pdm

### Migration Steps

#### 1. Export Dependencies
```bash
# PDM uses pyproject.toml which is mostly compatible
# Check for pdm-specific configuration
```

#### 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Update pyproject.toml
Remove pdm-specific sections:

```toml
# Remove or comment out
# [tool.pdm]
# [tool.pdm.dev-dependencies]
```

#### 4. Create Environment
```bash
uv venv
source .venv/bin/activate
uv sync
```

### PDM to uv Command Mapping

| PDM Command | uv Command | Notes |
|-------------|------------|-------|
| `pdm install` | `uv sync` | |
| `pdm add package` | `uv add package` | |
| `pdm remove package` | `uv remove package` | |
| `pdm update` | `uv update` | |
| `pdm run python` | `uv run python` | |
| `pdm list` | `uv pip list` | |

## Migrating from hatch

### Migration Steps

#### 1. Export Dependencies
Hatch uses pyproject.toml with different structure:

```toml
# Before (Hatch)
[project]
dependencies = [...]

# After (uv/standard)
[project]
dependencies = [...]
```

#### 2. Convert Optional Dependencies
```toml
# Hatch style
[tool.hatch.envs.default]
dependencies = [...]

# Standard style
[project.optional-dependencies]
dev = [...]
test = [...]
```

#### 3. Install and Test
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Enterprise Migration Strategy

### Phase 1: Assessment
```bash
# Inventory existing environments
# - List all projects
# - Document current tooling
# - Identify critical dependencies
# - Test uv compatibility

# Create migration matrix
| Project | Current Tool | Priority | Complexity | Target Date |
|---------|-------------|----------|------------|-------------|
| api-service | poetry | High | Medium | 2024-Q1 |
| data-pipeline | pip/venv | Medium | Low | 2024-Q2 |
```

### Phase 2: Pilot Migration
```bash
# Select low-risk project
# Document process and issues
# Create migration playbook
# Train team members
```

### Phase 3: Tooling and Automation
```bash
# Create migration scripts
# Update CI/CD pipelines
# Create validation tools
# Update developer documentation
```

### Phase 4: Full Migration
```bash
# Migrate remaining projects
# Monitor for issues
# Gather feedback
# Optimize process
```

### Phase 5: Optimization
```bash
# Implement best practices
# Set up monitoring
# Regular maintenance
# Training and documentation
```

## Migration Validation Checklist

### Pre-Migration
- [ ] Backup current environment
- [ ] Document current package versions
- [ ] Test critical functionality
- [ ] Create rollback plan

### During Migration
- [ ] Install uv successfully
- [ ] Create new environment
- [ ] Install all dependencies
- [ ] Verify package versions
- [ ] Test imports
- [ ] Run test suite
- [ ] Test application functionality

### Post-Migration
- [ ] Update documentation
- [ ] Update CI/CD pipelines
- [ ] Update Dockerfiles
- [ ] Train team members
- [ ] Monitor for issues
- [ ] Schedule regular updates

## Troubleshooting Migration

### Common Issues

#### 1. Missing Dependencies
```bash
# Check if package exists on PyPI
curl -s "https://pypi.org/pypi/package-name/json" | jq .info.version

# Search for alternative package names
uv add "package-name[extra]"  # Try with extras
uv add "package-name>=1.0.0,<2.0.0"  # Try version range
```

#### 2. Build Failures
```bash
# Install build dependencies
# Linux:
sudo apt-get install python3-dev build-essential

# Try binary wheels
uv add package --only-binary :all:

# Try different version
uv add "package<problematic-version"
```

#### 3. Performance Issues
```bash
# Enable parallel downloads
uv config set download.concurrency 20

# Use cache effectively
uv cache info
uv cache prune

# Consider pre-compiling requirements
uv pip compile requirements.in -o requirements.txt
```

#### 4. Team Coordination
- Create migration guide for team
- Schedule migration during low-activity periods
- Have rollback plan ready
- Communicate changes clearly

## Migration Success Metrics

### Quantitative
- **Build time reduction**: Measure before/after
- **Installation time**: Time to install dependencies
- **Cache hit rate**: Percentage of cache hits
- **Dependency resolution time**: Time to resolve dependencies
- **Memory usage**: Peak memory during installation

### Qualitative
- **Developer satisfaction**: Survey team members
- **Error rate reduction**: Fewer installation failures
- **Documentation clarity**: Ease of following new process
- **Tool reliability**: Fewer workarounds needed
- **Community support**: Availability of help/resources

## Rollback Plan

### When to Roll Back
- Critical functionality broken
- Performance regression
- Team productivity impacted
- Security concerns

### Rollback Steps
```bash
# 1. Restore backup
cp pyproject.toml.backup pyproject.toml
cp requirements.txt.backup requirements.txt

# 2. Recreate old environment
# For pip/venv:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For poetry:
curl -sSL https://install.python-poetry.org | python3 -
poetry install

# 3. Verify functionality
pytest
python -c "import critical_module; print('OK')"

# 4. Document issues for future migration attempt
```

### Partial Rollback
If only some aspects need rollback:

```bash
# Keep uv but use old dependency resolution
uv sync --resolution=lowest-direct

# Or use exact versions from backup
uv add -r requirements.txt.backup
```

## Best Practices for Smooth Migration

### 1. Test Thoroughly
```bash
# Create comprehensive test suite
uv run pytest --cov

# Test edge cases
uv run python -m test_edge_cases

# Performance testing
time uv sync
```

### 2. Document Everything
```markdown
# Migration Documentation

## Changes Made
1. Installed uv
2. Created .venv with Python 3.11
3. Migrated dependencies
4. Updated CI/CD
5. Updated documentation

## Issues Encountered
1. Package X required version pinning
2. Build dependency Y needed installation
3. Test Z needed adjustment

## Verification Steps
1. Run test suite: `uv run pytest`
2. Check imports: `uv run python -c "import all_modules"`
3. Performance check: `time uv sync`
```

### 3. Train the Team
- Workshop on uv basics
- Migration guide walkthrough
- Troubleshooting session
- Q&A with early adopters

### 4. Monitor After Migration
```bash
# Set up monitoring
# - Installation success rate
# - Build times
# - Error frequency
# - Developer feedback

# Regular check-ins for first month
# Weekly: Check for issues
# Monthly: Gather feedback
# Quarterly: Review metrics
```

### 5. Continuous Improvement
```bash
# Collect feedback
# Update documentation
# Optimize workflows
# Share learnings
```