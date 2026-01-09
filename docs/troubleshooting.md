# Troubleshooting Guide

This guide provides solutions to common issues encountered when setting up and running the YouTube-SC project. If you encounter problems not covered here, please check the [GitHub Issues](https://github.com/yourusername/Youtube-SC/issues) or contact the project maintainers.

## Environment Setup Issues

### Virtual Environment Problems

#### 1. Virtual Environment Activation Failure
**Symptoms**: `source .venv/bin/activate` or `.venv\Scripts\activate` fails
**Solutions**:
```bash
# Check if .venv directory exists
ls -la .venv/

# If missing, recreate environment
# Windows
setup-environment.bat

# Linux/Mac
./setup-environment.sh

# Manual creation
python -m venv .venv
```

#### 2. Virtual Environment Not Activated
**Symptoms**: Package import errors, Python uses system installation
**Verification**:
```bash
# Check Python path
which python  # Linux/Mac
where python  # Windows

# Should show path containing .venv
# Example: /path/to/Youtube-SC/.venv/bin/python

# Check terminal prompt (should show (.venv))
echo $PS1  # Might show virtual environment indicator
```

**Solutions**:
```bash
# Explicit activation
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Force reactivation
deactivate
source .venv/bin/activate
```

### Package Installation Issues

#### 3. Import Errors for Specific Packages
**Symptoms**: `ModuleNotFoundError` for pandas, torch, transformers, etc.
**Solutions**:
```bash
# Reactivate virtual environment first
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Reinstall all dependencies
uv pip install -r requirements.txt --force-reinstall

# Install specific missing package
uv install package_name
```

#### 4. Package Version Conflicts
**Symptoms**: `ImportError` due to version mismatches
**Solutions**:
```bash
# Check installed versions
uv pip list

# Install exact versions from requirements.txt
uv pip install -r requirements.txt --force-reinstall

# Create fresh environment
rm -rf .venv
./setup-environment.sh  # or setup-environment.bat
```

#### 5. Network Issues During Installation
**Symptoms**: Timeout errors, connection refused
**Solutions**:
```bash
# Use mirrors for faster downloads (China)
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Increase timeout
uv pip install -r requirements.txt --timeout 100

# Offline installation (if packages downloaded locally)
uv pip install --no-index --find-links=/path/to/packages -r requirements.txt
```

## Data-Related Issues

### 6. Missing Data Files
**Symptoms**: `FileNotFoundError` for `combined_comments.xlsx`
**Solutions**:
```bash
# Check data directory
ls data/

# Expected: combined_comments.xlsx should exist
# If missing:
# 1. Contact project maintainers for dataset access
# 2. Place file in data/ directory
# 3. Run data preparation
cd sentiment_classification_ML
python main.py --prepare
```

### 7. Data File Permission Issues
**Symptoms**: Permission denied errors when reading/writing files
**Solutions**:
```bash
# Check permissions
ls -la data/combined_comments.xlsx

# Fix permissions (Linux/Mac)
chmod 644 data/combined_comments.xlsx

# Run as administrator (Windows)
# Right-click terminal, "Run as administrator"
```

### 8. Encoding Problems (Chinese Text)
**Symptoms**: `UnicodeDecodeError`, GBK errors on Windows
**Solutions**:
Add encoding wrapper to Python scripts:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

Or specify encoding when reading files:
```python
import pandas as pd
df = pd.read_excel('data/combined_comments.xlsx', engine='openpyxl')
# or for CSV files
df = pd.read_csv('file.csv', encoding='utf-8')
```

## Machine Learning Module Issues

### 9. Word2Vec Download Issues
**Symptoms**: `ConnectionError` downloading Google News Word2Vec embeddings
**Solutions**:
```bash
# Use TF-IDF features instead (recommended)
cd sentiment_classification_ML
python main.py --train --model svm --type ovo --ngram 1

# Manual download alternative
# 1. Download from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
# 2. Place in: data/external/word2vec/GoogleNews-vectors-negative300.bin.gz
# 3. Run setup script
python main.py --setup-word2vec --local-path data/external/word2vec/GoogleNews-vectors-negative300.bin.gz

# Use smaller embeddings
python main.py --train --model svm --feature glove --embedding-dim 100
```

### 10. Memory Errors During Training
**Symptoms**: `MemoryError`, process killed, slow performance
**Solutions**:
```bash
# Reduce batch sizes
cd sentiment_classification_ML
python main.py --train --batch-size 256  # Default is 512

# Use sparse matrices for TF-IDF
# Already implemented in code

# Process in chunks
python main.py --train --chunk-size 1000

# Monitor memory usage
# Linux/Mac:
top
# Windows: Task Manager
```

### 11. Model Training Too Slow
**Solutions**:
```bash
# Use smaller n-gram range
python main.py --train --ngram 1  # Instead of (1,3)

# Reduce feature dimensions
python main.py --train --max-features 5000

# Use simpler model
python main.py --train --model nb  # Naive Bayes is faster than SVM
```

## BERT Module Issues

### 12. CUDA/GPU Issues
**Symptoms**: `CUDA out of memory`, GPU not detected
**Solutions**:
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Reduce batch size
cd sentiment_classification_Bert/code
python main.py --train_batch_size 16 --valid_batch_size 16

# Use CPU instead
python main.py --device cpu

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

### 13. BERT Model Download Issues
**Symptoms**: `ConnectionError` downloading `bert-base-uncased`
**Solutions**:
```bash
# Use offline mode if model already downloaded
# Model cache location: ~/.cache/huggingface/hub
# Copy to project directory
mkdir -p models/bert-base-uncased
cp -r ~/.cache/huggingface/hub/models--bert-base-uncased/* models/bert-base-uncased/

# Specify local path
python main.py --model-path models/bert-base-uncased

# Use different model
python main.py --model-name distilbert-base-uncased  # Smaller, faster
```

### 14. Training Divergence or NaN Loss
**Symptoms**: Loss becomes NaN, predictions are random
**Solutions**:
```bash
# Reduce learning rate
python main.py --learning_rate 1e-5

# Add gradient clipping
python main.py --max_grad_norm 1.0

# Use smaller batch size
python main.py --train_batch_size 16

# Check data for NaN values
python check_data.py --file data/train.csv
```

## General Python Issues

### 15. Python Version Issues
**Symptoms**: Syntax errors, incompatible libraries
**Solutions**:
```bash
# Check Python version
python --version
# Should be 3.11.x

# Use correct Python executable
# Ensure using virtual environment Python
.venv/bin/python --version  # Linux/Mac
.venv\Scripts\python --version  # Windows
```

### 16. Path Issues on Windows
**Symptoms**: Backslash vs forward slash issues, path length limits
**Solutions**:
```bash
# Use raw strings for Windows paths in Python
path = r"C:\Users\name\Youtube-SC\data\file.xlsx"

# Use pathlib for cross-platform compatibility
from pathlib import Path
data_path = Path("data") / "combined_comments.xlsx"

# Enable long path support (Windows)
# Run as administrator in PowerShell:
# Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

### 17. Jupyter Notebook Issues
**Symptoms**: Cannot import packages in Jupyter
**Solutions**:
```bash
# Install ipykernel in virtual environment
uv install ipykernel

# Add virtual environment to Jupyter
python -m ipykernel install --user --name=youtube-sc --display-name="Python (YouTube-SC)"

# Launch Jupyter from activated environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
jupyter notebook
```

## Performance Optimization

### 18. Slow Text Processing
**Solutions**:
```bash
# Use multiprocessing
cd sentiment_classification_ML
python main.py --n-jobs 4  # Use 4 CPU cores

# Cache processed data
# Already implemented - check data/processed/ directory

# Use optimized libraries
# Ensure using optimized NumPy (MKL/OpenBLAS)
python -c "import numpy; numpy.show_config()"
```

### 19. Large Memory Usage
**Solutions**:
```bash
# Use data streaming for large files
python main.py --stream --chunk-size 1000

# Enable garbage collection
import gc
gc.collect()

# Use memory-mapped files for large arrays
import numpy as np
arr = np.load('large_array.npy', mmap_mode='r')
```

### 20. Disk Space Issues
**Symptoms**: Out of disk space from model files or results
**Solutions**:
```bash
# Clean temporary files
# Windows
del /s /q *.pyc
for /d /r . %i in (__pycache__) do @if exist "%i" rmdir /s /q "%i"

# Linux/Mac
find . -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove old model checkpoints
find . -name "*.pkl" -mtime +7 -delete  # Older than 7 days
find . -name "*.pt" -mtime +7 -delete
find results/ -name "*.csv" -mtime +30 -delete  # Older than 30 days
```

## Operating System Specific Issues

### Windows Specific Issues

#### 21. Script Execution Policy
**Symptoms**: `File cannot be loaded because running scripts is disabled`
**Solutions**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run scripts with bypass
powershell -ExecutionPolicy Bypass -File setup-environment.bat
```

#### 22. Line Ending Issues
**Symptoms**: `^M` characters in scripts, syntax errors
**Solutions**:
```bash
# Convert line endings
# Using dos2unix (install via Chocolatey or manually)
dos2unix setup-environment.sh

# Using PowerShell
Get-Content -Path "setup-environment.sh" | Set-Content -NoNewline -Encoding UTF8 "setup-environment-unix.sh"

# Using Python
python -c "open('fixed.sh', 'w').write(open('original.sh').read().replace('\r\n', '\n'))"
```

### Linux/Mac Specific Issues

#### 23. Permission Denied for Scripts
**Symptoms**: `Permission denied` when running `.sh` scripts
**Solutions**:
```bash
# Make scripts executable
chmod +x setup-environment.sh
chmod +x scripts/*.sh

# Run with bash explicitly
bash setup-environment.sh
```

#### 24. Library Linking Issues
**Symptoms**: `libopenblas.so.0: cannot open shared object file`
**Solutions**:
```bash
# Install system libraries
# Ubuntu/Debian
sudo apt-get install libopenblas-dev libomp-dev

# CentOS/RHEL
sudo yum install openblas-devel

# Mac with Homebrew
brew install openblas
```

## Development Tools Issues

### 25. Git Issues
**Symptoms**: Large files in repository, merge conflicts
**Solutions**:
```bash
# Clean git history of large files
git filter-branch --tree-filter 'rm -f large_file.pkl' HEAD

# Resolve merge conflicts
git mergetool

# Update git configuration for large repos
git config --global core.compression 9
git config --global core.packedGitLimit 512m
git config --global core.packedGitWindowSize 512m
```

### 26. IDE/Editor Issues
**Symptoms**: Import errors in IDE but works in terminal
**Solutions**:
1. **VS Code**: Select Python interpreter from virtual environment
   - `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv/bin/python`
2. **PyCharm**: Configure project interpreter
   - File → Settings → Project → Python Interpreter → Add → Existing environment → Select `.venv/bin/python`
3. **Jupyter**: Install kernel from virtual environment (see section 17)

## Getting Further Help

### Diagnostic Information
When reporting issues, include:
```bash
# System information
python collect_diagnostics.py

# Or manually collect:
python --version
pip list  # or uv pip list
echo $PATH  # Linux/Mac
echo %PATH%  # Windows
```

### Contact Information
1. **GitHub Issues**: [Project Issues](https://github.com/yourusername/Youtube-SC/issues)
2. **Documentation**: Check relevant `.md` files in `docs/` directory
3. **Module READMEs**: Each module has its own `README.md`

### Debug Mode
Run modules with debug flags for more information:
```bash
cd sentiment_classification_ML
python main.py --verbose --debug --log-file debug.log

cd ../sentiment_classification_Bert/code
python main.py --debug --log-level DEBUG
```

## Prevention Tips

### Regular Maintenance
```bash
# Update packages regularly
uv update

# Clean cache files
uv cache clean

# Backup important results
tar -czf results_backup_$(date +%Y%m%d).tar.gz results/
```

### Environment Management
```bash
# Create environment snapshot
uv pip freeze > requirements_frozen.txt

# Recreate environment from snapshot
uv pip install -r requirements_frozen.txt

# Use Docker for consistent environment (optional)
docker build -t youtube-sc .
docker run -it youtube-sc
```

This troubleshooting guide covers the most common issues. For persistent problems, please open an issue on GitHub with detailed error messages and system information.