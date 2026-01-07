# YouTube-SC Project Guide

## Project Overview
- **Project Name**: YouTube-SC (YouTube Sentiment and Clustering Analysis)
- **Objective**: Analyze YouTube delivery robot comments through sentiment classification, clustering analysis, topic modeling, and text statistics.
- **Primary Modules**:
  - sdr_clustering_analysis: Clustering analysis of comments
  - sentiment_classification_ML: Machine learning-based sentiment classification
  - sentiment_classification_Bert: Deep learning-based sentiment classification using BERT
  - topic_modeling: Topic modeling analysis
  - text_statistics: Text statistical analysis

## Technology Stack
- **Python**: 3.11
- **Machine Learning**: scikit-learn, gensim, nltk
- **Deep Learning**: PyTorch, Transformers, Sentence-Transformers
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud

## Code Style & Conventions
- **Naming Conventions**:
  - Variables and functions: snake_case
  - Class names: CamelCase
  - Constants: UPPER_CASE
- **Documentation**: Chinese docstrings with triple double quotes, including parameter descriptions and return values
- **Import Order**:
  1. Standard library imports
  2. Third-party library imports
  3. Local module imports
- **Configuration**: Each module has independent config.py files for managing paths, parameters, and hyperparameters
- **Error Handling**: try-except blocks for expected exceptions with error logging
- **Testing**: Each module includes test scripts with assertions

## Project Structure
```
Youtube-SC/
├── Code/                         # Functional modules
│   ├── sdr_clustering_analysis/  # Clustering analysis
│   ├── sentiment_classification_ML/      # ML sentiment classification
│   ├── sentiment_classification_Bert/    # BERT sentiment classification
│   ├── topic_modeling/           # Topic modeling
│   └── text_statistics/          # Text statistics (contains Chinese directory names)
├── data/                         # Shared data directory
│   ├── Comments/                 # Comment data
│   │   ├── SDR_10_50_comments/   # Comments for SDR 10-50
│   │   ├── SDR_50_100_comments/  # Comments for SDR 50-100
│   │   └── SDR_100_1000_comments/ # Comments for SDR 100-1000
│   └── Video_statistics/         # Video statistics data
├── .claude/                      # Claude Code configuration
│   ├── task-board.md             # Project task management
│   ├── bug-log.md                # Bug tracking and debugging
│   ├── SKILLS_README.md          # Skills documentation
│   ├── requirements.txt          # Project dependencies
│   ├── setup-environment.bat     # Windows environment setup
│   ├── setup-environment.sh      # Linux/Mac environment setup
│   └── skills/                   # Custom skills
│       ├── update-task-board/    # Task management skill
│       ├── log-debug-issue/      # Bug logging skill
│       └── manage-python-env/    # Python environment management skill
└── .serena/                     # Serena AI configuration
```

## Entry Points
Each module has its own main entry point:
- `Code/sdr_clustering_analysis/main.py`
- `Code/sentiment_classification_ML/main.py`
- `Code/sentiment_classification_Bert/code/main.py`
- `Code/topic_modeling/topic_modeling_analysis.py`

## Data Flow
Raw comment data (`data/Comments/combined_comments.xlsx`) is processed by each module, with outputs saved to respective module's `results/` directories.

## Frequently Used Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize NLTK data (if needed)
python -m nltk.downloader stopwords wordnet punkt

# Windows environment setup
.claude/setup-environment.bat

# Linux/Mac environment setup
.claude/setup-environment.sh
```

### Module Execution
```bash
# Clustering analysis
cd Code/sdr_clustering_analysis && python main.py

# ML sentiment classification
cd Code/sentiment_classification_ML && python main.py --help

# BERT sentiment classification
cd Code/sentiment_classification_Bert/code && python main.py --help

# Topic modeling
cd Code/topic_modeling && python topic_modeling_analysis.py
```

### Project Management
```bash
# Clean Python cache (Windows)
for /d /r Code %i in (__pycache__) do @if exist "%i" rmdir /s /q "%i"

# Count Python files
dir /s /b Code\*.py | find /c ".py"
```

### Custom Claude Skills
This project includes three custom skills:
1. **update-task-board**: Task management using markdown files
2. **log-debug-issue**: Bug and issue tracking system
3. **manage-python-env**: Python virtual environment management using uv

## Key Configuration Files
- **requirements.txt**: Project dependencies in `.claude/` directory
- **task-board.md**: Project task management in `.claude/` directory
- **bug-log.md**: Bug tracking in `.claude/` directory
- **project.yml**: Serena configuration in `.serena/` directory

## Notes
- **Chinese Content**: Some directories and documentation contain Chinese text
- **Encoding**: UTF-8 encoding used throughout
- **Platform Compatibility**: Watch for Windows GBK encoding issues with Unicode characters (documented in bug-log.md)
- **Module Independence**: Each module operates independently with its own data files

## Recent Changes
- **Refactoring Completed**: Old code deleted, new code cleaned (2026-01-06)
- **Custom Skills Created**: Three management skills added for task tracking, bug logging, and environment management
- **Unified Dependencies**: Single requirements.txt file created for all modules

## Useful References
- See `.serena/memories/` for original project documentation in Chinese
- Check `.claude/task-board.md` for detailed refactoring progress
- Review `.claude/bug-log.md` for troubleshooting common issues