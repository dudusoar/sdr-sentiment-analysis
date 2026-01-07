# BERT Sentiment Classification Module

## Overview
This module provides BERT-based sentiment classification for YouTube delivery robot comments. It uses BERT with additional LSTM/GRU layers and supports three sentiment classes: posi, neg, neutral.

## Project Structure

```
sentiment_classification_Bert/
├── code/                          # Main source code
│   ├── main.py                    # Primary training and evaluation
│   ├── test.py                    # Testing script
│   ├── predict.py                 # Prediction script
│   ├── train.sh                   # Training shell script (Linux)
│   ├── preprocess/                # Data preprocessing
│   │   └── processor.py           # Dataset processor
│   ├── utils/                     # Utility functions
│   │   ├── trainer.py             # Training utilities
│   │   ├── model_utils.py         # BERT model implementations
│   │   ├── baseloss.py            # Custom loss functions
│   │   └── function_utils.py      # Helper functions
├── data/                          # Data directory
│   ├── comments_new/              # Processed comment data
│   │   ├── train.csv              # Training data (3676 samples)
│   │   ├── val.csv                # Validation data (460 samples)
│   │   └── test.csv               # Test data (459 samples)
└── user_data/                     # User-generated data
    ├── checkpoint/                # Model checkpoints
    └── results/                   # Results directory
```

## Data Format

Training data CSV files contain:
- `pure_text`: Preprocessed text
- `orginial_text`: Original tokenized text (JSON-like list)
- `len`: Text length
- `label`: Sentiment label (0: posi, 1: neg, 2: neutral)

## Installation

1. **Install dependencies** (use the project's unified requirements.txt):
   ```bash
   uv pip install -r ../requirements.txt
   ```

2. **Create necessary directories**:
   ```bash
   mkdir -p user_data/checkpoint user_data/results
   ```

## Usage

### Basic Training
```bash
cd sentiment_classification_Bert/code
python main.py --epoches 10 --train_batch_size 32 --valid_batch_size 32
```

### Training with Custom Parameters
```bash
python main.py \
  --epoches 20 \
  --train_batch_size 64 \
  --valid_batch_size 64 \
  --learning_rate 3e-5 \
  --max_seq_length 128 \
  --early_stopping 5
```

### Testing (requires trained model)
```bash
python test.py --test_batch_size 64
```

### Prediction
```bash
python predict.py
```

## Configuration Parameters

### Data Parameters
- `--train_dir`, `--val_dir`, `--test_dir`: Paths to data files
- `--max_seq_length`: Maximum sequence length (default: 120)

### Model Parameters
- `--MODEL_NAME`: Model identifier (default: "bert-base-wwm")
- `--learning_rate`: Initial learning rate (default: 5e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--enable_mdrop`: Enable multi-dropout (default: False)

### Training Parameters
- `--epoches`: Number of training epochs (default: 30)
- `--train_batch_size`, `--valid_batch_size`, `--test_batch_size`: Batch sizes
- `--early_stopping`: Early stopping patience (default: 3)
- `--grad_accumulate_nums`: Gradient accumulation steps (default: 1)

### Path Parameters
- `--save_dir_curr`: Checkpoint save directory
- `--results_excel_dir`: Results directory

### State Parameters
- `--is_train`: Enable training ("yes"/"no", default: "yes")
- `--is_predict`: Enable prediction ("yes"/"no", default: "yes")

## Model Architecture

The model uses a hybrid architecture:
1. **BERT Encoder**: Pretrained BERT model for text encoding
2. **LSTM Layer**: Bidirectional LSTM for sequence modeling
3. **GRU Layer**: Bidirectional GRU for additional sequence processing
4. **Classifier**: Linear layer for sentiment classification

## Notes

### Platform Compatibility
- Shell scripts (`train.sh`) are for Linux/macOS
- For Windows, use Python commands directly or Git Bash
- Chinese text encoding may display as garbled in Windows console but doesn't affect functionality

### Model Selection
- Currently configured to use `bert-base-uncased` from HuggingFace (English BERT model)
- Model will be automatically downloaded on first run
- Using English model because the comment data is in English

### Performance Considerations
- Training is memory-intensive; adjust batch size based on available GPU memory
- Consider using gradient accumulation for larger effective batch sizes
- Early stopping helps prevent overfitting

### Label Mapping Note
**Important**: Due to historical design decisions, the label mapping has been changed:
- `0` → `'posi'` (positive sentiment)
- `1` → `'neg'` (negative sentiment)
- `2` → `'neutral'` (neutral sentiment)

This is different from the original mapping and represents a design simplification. Existing trained models may need retraining with the new label mapping.

## Troubleshooting

### Common Issues

1. **Import errors**:
   - Ensure all dependencies are installed from `../requirements.txt`
   - Check Python version (requires 3.11+, 3.11.12 recommended)

2. **CUDA/GPU issues**:
   - Comment out GPU-specific lines if running on CPU
   - Ensure PyTorch is installed with CUDA support if using GPU

3. **Model download errors**:
   - Check internet connection
   - The model will be cached locally after first download

4. **Checkpoint loading errors**:
   - Ensure `user_data/checkpoint` directory exists
   - Train a model first before testing/prediction

## Fixed Issues

### Critical Issues Fixed:

1. **Hardcoded Linux Paths** (High Severity)
   - **Problem**: `MODEL_DIR` in `main.py`, `test.py`, and `predict.py` contained hardcoded Linux paths
   - **Fix**: Changed to use HuggingFace model names (`bert-base-uncased` for English text)
   - **Files**: `main.py:42-46`, `test.py:30-32`, `predict.py:15-17`

2. **Inconsistent Label Mapping** (Medium Severity)
   - **Problem**: Duplicate terms for "negative" (different Chinese terms mapped to different IDs)
   - **Fix**: Standardized to three-class sentiment: posi, neg, neutral
   - **Files**: `main.py:105-107`, `test.py:111-113`, `predict.py:68-70`

3. **GPU Hardcoding** (Medium Severity)
   - **Problem**: `os.environ['CUDA_VISIBLE_DEVICES'] = '2'` hardcoded to specific GPU
   - **Fix**: Commented out, allowing automatic GPU detection or CPU fallback
   - **Files**: `main.py:6`

4. **AdamW Compatibility** (Medium Severity)
   - **Problem**: `transformers.AdamW` import error with newer Transformers versions
   - **Fix**: Changed to `torch.optim.AdamW` and removed `correct_bias` parameter
   - **Files**: `trainer.py:12-13`, `trainer.py:57`

5. **Uninitialized Variables** (Low Severity)
   - **Problem**: `best_score` and `test_score` not initialized when `is_train='no'`
   - **Fix**: Added initialization before conditional blocks
   - **Files**: `main.py:147-149`

   
## Development Context

**Important**: This codebase was originally developed in **2023** and has not been actively maintained or updated since its initial completion. The recent modifications listed above were made to address critical compatibility issues and language inconsistencies, but the core architecture and implementation remain from the original 2023 development.

### Code Status
- **Original Development**: 2023
- **Last Major Update**: 2023 (initial completion)
- **Recent Maintenance**: Minor fixes for compatibility and language standardization only
- **Core Architecture**: Unchanged since 2023 development

## Known Issues

### GradScaler CPU Compatibility (High Severity)
- **Problem**: `GradScaler()` in `trainer.py:139` and `trainer_orign.py:118` is unconditionally created without checking for CUDA availability
- **Impact**: Code fails when running on CPU-only systems, as `GradScaler` requires CUDA for mixed precision training
- **Root Cause**: Original 2023 implementation assumed GPU availability and didn't include device checks
- **Files**: `trainer.py:139`, `trainer_orign.py:118`
- **Status**: **Not fixed** - remains as originally implemented in 2023

### Background
The `GradScaler` class from `torch.cuda.amp` is part of PyTorch's Automatic Mixed Precision (AMP) training, which requires CUDA-capable GPUs. When running on CPU-only systems, `GradScaler()` instantiation will fail with CUDA-related errors.

### Workaround
1. Run the code on a system with CUDA-capable GPU
2. Or manually modify the code to conditionally use `GradScaler` only when `device.type == 'cuda'`

## References

- Original code was developed with Chinese labels but analyzes English YouTube comments
- Uses HuggingFace Transformers library
- Incorporates custom loss functions from research papers

## Recent Changes (2026-01-07)

**Note**: These are maintenance fixes for compatibility and language standardization only. The core code architecture remains unchanged from the original 2023 implementation.

1. Fixed hardcoded Linux paths to use HuggingFace model names
2. Standardized label mapping to three-class sentiment
3. Fixed AdamW compatibility with newer Transformers versions
4. Added variable initialization to prevent UnboundLocalError
5. Commented out GPU hardcoding for better portability
6. Changed model from `bert-base-chinese` to `bert-base-uncased` (English model for English text)

## Next Steps

1. Add Windows batch script equivalents
2. Create configuration file for easier parameter management
3. Add data preprocessing scripts
4. Implement model evaluation metrics
5. Add support for English language models