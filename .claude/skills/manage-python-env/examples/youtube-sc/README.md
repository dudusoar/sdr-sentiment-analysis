# YouTube-SC Project Examples

This directory contains real-world examples of Python environment management configuration files from the YouTube-SC project.

## Files

### 1. `requirements.txt`
- **Purpose**: Project dependency specifications
- **Contents**: All dependencies needed for the YouTube-SC project including:
  - Core data processing: pandas, numpy
  - Machine learning: scikit-learn, gensim, nltk
  - Deep learning: PyTorch, transformers, sentence-transformers
  - Visualization: matplotlib, seaborn, wordcloud
- **Usage**: `pip install -r requirements.txt` or `uv sync`

### 2. `setup-environment.bat`
- **Purpose**: Windows environment setup script using uv
- **Features**:
  - Checks Python installation
  - Installs uv if not present
  - Creates Python 3.11 virtual environment
  - Activates environment
  - Installs dependencies from requirements.txt
  - Verifies installation of key packages
- **Usage**: Run from project root directory

### 3. `setup-environment.sh`
- **Purpose**: Linux/macOS environment setup script using uv
- **Features**: Same functionality as Windows version but for Unix-based systems
- **Usage**: Make executable and run from project root directory

## Project Context

**YouTube-SC** is a YouTube delivery robot comment analysis system with five main modules:
1. **sdr_clustering_analysis** - Clustering analysis of comments
2. **sentiment_classification_ML** - Machine learning-based sentiment classification
3. **sentiment_classification_Bert** - Deep learning-based sentiment classification using BERT
4. **topic_modeling** - Topic modeling analysis
5. **text_statistics** - Text statistical analysis

## Key Design Patterns

### Dependency Management
- Single `requirements.txt` for all modules
- Clear separation between core and optional packages
- Version ranges for development flexibility
- Deep learning packages marked as optional

### Environment Scripts
- Cross-platform compatibility (Windows/Linux/macOS)
- Progressive enhancement approach
- Error handling and fallback mechanisms
- Installation verification

### uv Integration
- Modern Python package management
- Fast dependency resolution
- Unified environment management
- Cross-platform consistency

## How These Files Relate to the Skill

These files demonstrate practical application of the `manage-python-env` skill concepts:

1. **Real-world requirements file**: Shows how to structure dependencies for a multi-module ML project
2. **Production-ready setup scripts**: Complete environment automation examples
3. **Error handling patterns**: Demonstrates robust script design
4. **Cross-platform compatibility**: Shows how to support Windows and Unix systems

## Usage Notes

- These files are specific to the YouTube-SC project but serve as templates
- Modify package versions based on your project needs
- Adjust Python version requirements as needed
- Update module-specific commands for your project structure

## See Also

- [Skill Documentation](../SKILL.md)
- [uv Documentation](../references/uv-docs.md)
- [Common Recipes](../references/recipes.md)
- [Troubleshooting Guide](../references/troubleshooting.md)