# YouTube-SC Git Workflow Examples

This directory contains git workflow examples specific to the YouTube-SC project.

## Project Structure
```
Youtube-SC/
├── Code/                         # Functional modules
│   ├── sdr_clustering_analysis/
│   ├── sentiment_classification_ML/
│   ├── sentiment_classification_Bert/
│   ├── topic_modeling/
│   └── text_statistics/
├── data/                         # Data files
└── .claude/                      # Configuration and skills
```

## Common Git Scenarios

### 1. Starting Work on a New Module Feature

```bash
# Update local develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/clustering-enhancement

# Make changes to clustering module
cd Code/sdr_clustering_analysis
# Edit files...

# Stage and commit changes
git add .
git commit -m "feat(clustering): add silhouette analysis for cluster validation"

# Continue development
git add .
git commit -m "feat(clustering): add visualization for silhouette scores"

# Push to remote
git push -u origin feature/clustering-enhancement
```

### 2. Fixing a Bug in Sentiment Analysis

```bash
# Create bugfix branch from main
git checkout main
git pull origin main
git checkout -b bugfix/sentiment-memory-issue

# Fix the bug
cd Code/sentiment_classification_ML
# Fix memory leak...

# Add test for the fix
git add .
git commit -m "fix(sentiment): resolve memory leak in large dataset processing"

git add .
git commit -m "test(sentiment): add regression test for memory fix"

# Push for code review
git push -u origin bugfix/sentiment-memory-issue
```

### 3. Preparing a Release

```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/1.1.0

# Update version information
# Update CHANGELOG.md, version files...

git add .
git commit -m "chore: prepare release 1.1.0"

# Final testing
# Run tests across all modules...

git add .
git commit -m "test: final release testing complete"

# Merge to main and tag
git checkout main
git merge --no-ff release/1.1.0
git tag -a v1.1.0 -m "Release 1.1.0"

# Merge back to develop
git checkout develop
git merge --no-ff release/1.1.0

# Push everything
git push origin main --tags
git push origin develop

# Cleanup
git branch -d release/1.1.0
git push origin --delete release/1.1.0
```

### 4. Working with Multiple Modules

```bash
# Start feature affecting multiple modules
git checkout -b feature/cross-module-integration

# Work on clustering module
cd Code/sdr_clustering_analysis
git add .
git commit -m "feat(clustering): update output format for integration"

# Work on sentiment module
cd ../sentiment_classification_ML
git add .
git commit -m "feat(sentiment): add clustering result input support"

# Work on integration script
cd ../..
git add .
git commit -m "feat(integration): add cross-module analysis pipeline"

# Push complete feature
git push -u origin feature/cross-module-integration
```

## Branch Naming Convention for YouTube-SC

### Feature Branches
```
feature/clustering-optimization
feature/bert-fine-tuning
feature/topic-visualization
feature/stats-export
```

### Bugfix Branches
```
bugfix/data-loading-error
bugfix/visualization-crash
bugfix/performance-issue
```

### Release Branches
```
release/1.0.0
release/1.1.0
release/2.0.0
```

### Hotfix Branches
```
hotfix/critical-security
hotfix/production-crash
```

## Commit Message Examples for YouTube-SC

### Module-Specific Commits
```bash
feat(clustering): add DBSCAN clustering algorithm
fix(sentiment): correct label encoding issue
docs(topic): update LDA parameter documentation
test(stats): add coverage for text frequency analysis
```

### Cross-Module Commits
```bash
feat(integration): connect clustering with sentiment analysis
refactor(common): extract shared utility functions
chore(deps): update all module dependencies
```

### Data-Related Commits
```bash
feat(data): add new comment dataset loader
fix(data): resolve encoding issues with Chinese comments
docs(data): update dataset documentation
```

## Git Configuration for YouTube-SC

### Recommended .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files (large files)
data/*.pkl
data/*.model
data/*.h5
data/*.hdf5

# Results
results/*.png
results/*.jpg
results/*.pdf

# Logs
*.log
```

### Useful Aliases
```bash
# Add to ~/.gitconfig
[alias]
    sc-status = "!f() { echo '=== YouTube-SC Status ==='; git status; }; f"
    sc-log = "log --oneline --graph --all -20"
    sc-branches = "branch -a --sort=-committerdate"
    sc-clean = "!f() { git branch --merged main | grep -v \"\\*\" | xargs -n 1 git branch -d; }; f"
```

## Collaborative Workflow

### 1. Feature Development with Team
```bash
# Clone repository
git clone https://github.com/username/Youtube-SC.git
cd Youtube-SC

# Create feature branch
git checkout -b feature/team-collaboration

# Make your changes
git add .
git commit -m "feat: initial implementation"

# Push and create PR
git push -u origin feature/team-collaboration
# Create Pull Request on GitHub

# Address review comments
git add .
git commit -m "fix: address review feedback"
git push origin feature/team-collaboration
```

### 2. Keeping Branch Updated
```bash
# Fetch latest changes
git fetch origin

# Rebase on develop
git rebase origin/develop

# Resolve conflicts if any
# Continue with git add, git rebase --continue

# Force push (history changed)
git push origin feature/team-collaboration --force
```

### 3. Code Review Process
```bash
# After PR approval, merge locally
git checkout develop
git pull origin develop
git merge --no-ff feature/team-collaboration

# Run final tests
# Push to remote
git push origin develop

# Cleanup
git branch -d feature/team-collaboration
git push origin --delete feature/team-collaboration
```

## Troubleshooting

### Common Issues

#### 1. Large Data Files in Commits
```bash
# Remove accidentally committed large files
git filter-branch --tree-filter 'rm -f data/large-file.pkl' HEAD

# Use git-lfs for large files
git lfs track "*.pkl"
git add .gitattributes
git add data/large-file.pkl
git commit -m "feat: add dataset with git-lfs"
```

#### 2. Module-Specific Changes Only
```bash
# Commit changes only in specific module
git add Code/sdr_clustering_analysis/
git commit -m "feat(clustering): update algorithm"

# Leave other changes unstaged
```

#### 3. Reverting Module Changes
```bash
# Revert changes in specific module
git checkout HEAD -- Code/sentiment_classification_ML/

# Or revert specific commit
git revert abc123 --no-commit
# Edit to keep other module changes
git commit -m "revert: undo sentiment module changes"
```

## Best Practices for YouTube-SC

1. **Module Isolation**: Keep commits focused on specific modules when possible
2. **Data Management**: Use git-lfs for large data files, keep raw data out of repo
3. **Testing**: Run module tests before committing
4. **Documentation**: Update module READMEs when making significant changes
5. **Dependencies**: Update requirements.txt when adding new packages
6. **Backwards Compatibility**: Consider impact on other modules when making changes