# Environment Setup

This guide provides complete instructions for setting up the development environment for the YouTube-SC project.

## Prerequisites
- **Python**: 3.11+ (3.11.12 recommended for optimal compatibility)
- **Dependency Management**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Git**: For cloning the repository
- **Operating System**: Windows, Linux, or macOS

## Environment Overview
- **Python Version**: 3.11.12
- **Virtual Environment Path**: `.venv/` (created automatically)
- **Dependency Management Tool**: uv
- **Number of Installed Packages**: 138
- **Environment Creation Date**: 2026-01-06

## Installation Methods

### Option 1: Using uv (Recommended)
uv provides faster dependency resolution and installation.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Youtube-SC.git
   cd Youtube-SC
   ```

2. **Set up environment using automated scripts**:
   ```bash
   # Windows
   setup-environment.bat

   # Linux/Mac
   ./setup-environment.sh
   ```

   These scripts will:
   - Create a virtual environment in `.venv/`
   - Install all dependencies from `requirements.txt`
   - Download necessary NLTK data
   - Set up project configuration

### Option 2: Manual Setup with pip
If you prefer using pip instead of uv:

1. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate (Windows)
   .venv\Scripts\activate

   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```bash
   python -m nltk.downloader stopwords wordnet punkt
   ```

## Virtual Environment Activation

After installation, you need to activate the virtual environment before running any project code:

### Windows
**PowerShell/CMD:**
```cmd
.venv\Scripts\activate
```

**Git Bash:**
```bash
source .venv/Scripts/activate
```

### Linux/Mac
```bash
source .venv/bin/activate
```

**Verification**: After activation, you should see `(.venv)` prefix in your terminal prompt.

## Verification

After setting up the environment, verify key packages are installed correctly:

```bash
# After activating environment, verify key packages
python -c "import pandas; print(f'✅ pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'✅ scikit-learn: {sklearn.__version__}')"
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'✅ transformers: {transformers.__version__}')"
python -c "import gensim; print(f'✅ gensim: {gensim.__version__}')"
```

## Dependency Management

Once the environment is set up, you can manage dependencies using the following commands:

### Using uv
**View installed packages:**
```bash
uv pip list
```

**Update packages:**
```bash
uv update
# or update specific package
uv update package_name
```

**Add new packages:**
```bash
uv add package_name
```

**Install from requirements.txt:**
```bash
uv pip install -r requirements.txt
```

### Using pip
**View installed packages:**
```bash
pip list
```

**Update packages:**
```bash
pip install --upgrade package_name
```

**Add new packages:**
```bash
pip install package_name
```

## Key Package Versions

The current environment includes these key packages with their versions:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation and analysis |
| numpy | 2.4.0 | Numerical computing |
| scikit-learn | 1.8.0 | Machine learning algorithms |
| torch | 2.9.1+cpu | Deep learning framework |
| transformers | 4.57.3 | Pre-trained transformer models |
| gensim | 4.4.0 | Topic modeling and word embeddings |
| nltk | 3.9.2 | Natural language processing |
| matplotlib | 3.10.8 | Data visualization |
| seaborn | 0.13.2 | Statistical data visualization |
| wordcloud | 1.9.5 | Word cloud generation |

## Environment Recreation

If you need to recreate the environment from scratch:

1. **Delete the existing virtual environment**:
   ```bash
   # Windows
   rmdir /s /q .venv

   # Linux/Mac
   rm -rf .venv
   ```

2. **Run the setup script again**:
   ```bash
   # Windows
   setup-environment.bat

   # Linux/Mac
   ./setup-environment.sh
   ```

## Notes
- **Always activate the virtual environment** before running any project code
- **All project modules should run within the activated environment**
- **For encoding issues** (GBK errors on Windows): Use UTF-8 encoding for files or add encoding wrapper in Python scripts:
  ```python
  import sys
  import io
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
  ```