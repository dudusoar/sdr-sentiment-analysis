#!/bin/bash
# YouTube-SC project environment setup script (Linux/macOS)
# Use uv to manage Python environment

echo "========================================"
echo "   YouTube-SC Project Environment Setup"
echo "========================================"
echo ""

# Checking Python
echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 not installed or not in PATH"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Check uv installation
echo ""
echo "Checking uv installation..."
if ! command -v uv &> /dev/null; then
    echo "uv not installed, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Re-check
    if ! command -v uv &> /dev/null; then
        echo "Error: uv installation failed"
        echo "Please install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
    echo "✅ uv installation successful"
else
    echo "✅ uv already installed"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv --python 3.11 .venv
if [ $? -ne 0 ]; then
    echo "Error: Creating virtual environment failed"
    exit 1
fi
echo "✅ virtual environment created successfully"

# Activate environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Activating virtual environment failed"
    exit 1
fi
echo "✅ virtual environment activated"

# Install dependencies
echo ""
echo "Installing project dependencies..."
uv sync
if [ $? -ne 0 ]; then
    echo "Warning: dependencies installation encountered problems"
    echo "Trying alternative..."
    
    # Try installing from requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt..."
        uv add -r requirements.txt
    else
        echo "Error: dependencies file not found"
        echo "Please install dependencies manually"
    fi
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import pandas; print(f'✅ pandas version: {pandas.__version__}')"
python -c "import sklearn; print(f'✅ scikit-learn version: {sklearn.__version__}')"
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch not installed (optional)"

echo ""
echo "========================================"
echo "    Environment setup completed!"
echo "========================================"
echo ""
echo "Available commands:"
echo "  - Run clustering analysis: cd Code/sdr_clustering_analysis && python main.py"
echo "  - Run ML sentiment classification: cd Code/sentiment_classification_ML && python main.py --help"
echo "  - Run BERT sentiment classification: cd Code/sentiment_classification_Bert/code && python main.py --help"
echo "  - Run topic modeling: cd Code/topic_modeling && python topic_modeling_analysis.py"
echo ""
echo "Next time, activate the environment first:"
echo "  source .venv/bin/activate"
echo ""