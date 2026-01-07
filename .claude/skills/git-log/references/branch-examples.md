# Branch Management Examples

## Branch Naming Conventions

### Standard Branch Types
```
main                    # Production-ready code
develop                 # Development integration
feature/<name>          # New features
bugfix/<name>           # Bug fixes
release/<version>       # Release preparation
hotfix/<name>           # Critical production fixes
```

### YouTube-SC Project Examples
```
main
develop
feature/clustering-enhancement
feature/bert-sentiment
bugfix/data-loading-issue
release/1.0.0
hotfix/critical-memory-leak
```

## Branch Creation Examples

### Feature Branch
```bash
# Create from develop branch
git checkout develop
git pull origin develop
git checkout -b feature/new-analysis-module

# Work on feature
git add .
git commit -m "feat: add initial analysis module structure"
git commit -m "feat: implement core analysis algorithms"
git commit -m "test: add unit tests for analysis module"

# Push to remote
git push -u origin feature/new-analysis-module
```

### Bugfix Branch
```bash
# Create from main for production bug
git checkout main
git pull origin main
git checkout -b bugfix/memory-leak-issue

# Fix the bug
git add .
git commit -m "fix: resolve memory leak in data processing"

# Test and push
git push -u origin bugfix/memory-leak-issue
```

### Release Branch
```bash
# Create from develop when feature complete
git checkout develop
git pull origin develop
git checkout -b release/1.2.0

# Prepare release
git commit -m "chore: update version to 1.2.0"
git commit -m "docs: update changelog for release"

# Merge to main and tag
git checkout main
git merge release/1.2.0
git tag -a v1.2.0 -m "Release 1.2.0"

# Merge back to develop
git checkout develop
git merge release/1.2.0
```

## Branch Workflow Examples

### Feature Development Workflow

#### 1. Start Feature Development
```bash
# Update local develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/sentiment-analysis
```

#### 2. Develop Feature
```bash
# Make changes and commit
git add .
git commit -m "feat(sentiment): add basic sentiment analysis"

# Continue development
git add .
git commit -m "feat(sentiment): add ML model integration"

# Add tests
git add .
git commit -m "test(sentiment): add unit tests"
```

#### 3. Keep Feature Updated
```bash
# Rebase on latest develop
git fetch origin
git rebase origin/develop

# Resolve conflicts if any
# Continue development
```

#### 4. Complete Feature
```bash
# Final commits
git add .
git commit -m "docs(sentiment): update documentation"

# Push to remote
git push -u origin feature/sentiment-analysis

# Create pull request to develop
```

### Hotfix Workflow

#### 1. Create Hotfix Branch
```bash
# From main (production)
git checkout main
git pull origin main
git checkout -b hotfix/critical-issue
```

#### 2. Fix the Issue
```bash
# Make minimal fix
git add .
git commit -m "fix: resolve critical data corruption issue"

# Add test for fix
git add .
git commit -m "test: add regression test for fix"
```

#### 3. Deploy Hotfix
```bash
# Merge to main
git checkout main
git merge hotfix/critical-issue

# Tag release
git tag -a v1.0.1 -m "Hotfix for critical issue"

# Merge to develop
git checkout develop
git merge hotfix/critical-issue

# Push everything
git push origin main --tags
git push origin develop
```

#### 4. Cleanup
```bash
# Delete hotfix branch
git branch -d hotfix/critical-issue
git push origin --delete hotfix/critical-issue
```

## Branch Management Commands

### Viewing Branches
```bash
# Local branches
git branch

# All branches (including remote)
git branch -a

# Branches with last commit
git branch -v

# Branches merged into current
git branch --merged

# Branches not merged into current
git branch --no-merged
```

### Branch Operations
```bash
# Create branch from specific commit
git branch new-feature abc123

# Create branch from tag
git branch release-candidate v1.0.0

# Move branch pointer (force)
git branch -f feature-branch develop

# Set upstream branch
git branch -u origin/feature-branch

# Delete merged branches
git branch --merged develop | grep -v "\*" | xargs -n 1 git branch -d
```

### Remote Branch Management
```bash
# Track remote branch
git checkout -b feature-branch origin/feature-branch

# Delete remote branch
git push origin --delete old-feature

# Prune stale remote tracking branches
git remote prune origin

# List remote branches
git ls-remote origin
```

## Merge Strategies

### Fast-Forward Merge
```bash
# When branch is linear descendant
git checkout main
git merge feature/simple

# Result: linear history, no merge commit
```

### No-Fast-Forward Merge
```bash
# Always create merge commit
git checkout main
git merge --no-ff feature/complex

# Result: explicit merge commit, preserves branch history
```

### Squash Merge
```bash
# Combine all feature commits into one
git checkout main
git merge --squash feature/multiple-commits
git commit -m "feat: add complete feature implementation"

# Result: clean history, single commit
```

## Rebase Examples

### Interactive Rebase
```bash
# Rebase last 5 commits
git rebase -i HEAD~5

# Common commands in rebase editor:
# pick - use commit
# reword - use commit, edit message
# edit - use commit, stop for amending
# squash - combine with previous commit
# fixup - like squash but discard message
# drop - remove commit
```

### Rebase Feature Branch
```bash
# Update feature branch with latest develop
git checkout feature/my-feature
git fetch origin
git rebase origin/develop

# If conflicts occur:
# 1. Resolve conflicts in files
# 2. git add <resolved-files>
# 3. git rebase --continue

# Force push (since history changed)
git push origin feature/my-feature --force
```

## Branch Cleanup

### Delete Local Branches
```bash
# Delete merged branches
git branch --merged main | grep -v "\*" | xargs -n 1 git branch -d

# Delete feature branches (interactive)
git branch | grep feature/ | xargs -n 1 git branch -d

# Force delete unmerged branches
git branch -D experimental-branch
```

### Delete Remote Branches
```bash
# Delete specific remote branch
git push origin --delete old-branch

# Delete multiple remote branches
git push origin --delete feature/old-feature bugfix/fixed-issue

# Cleanup after merged PRs
git fetch origin --prune
```

## GitHub Flow Example

### 1. Create Branch
```bash
git checkout -b feature/new-analysis develop
```

### 2. Add Commits
```bash
git add .
git commit -m "feat: add initial analysis logic"
git add .
git commit -m "test: add analysis tests"
```

### 3. Open Pull Request
```bash
git push -u origin feature/new-analysis
# Then create PR on GitHub
```

### 4. Address Review Comments
```bash
# Make requested changes
git add .
git commit -m "fix: address review comments"

# Update PR
git push origin feature/new-analysis
```

### 5. Merge PR
```bash
# On GitHub: Merge pull request
# Then locally:
git checkout develop
git pull origin develop

# Delete local branch
git branch -d feature/new-analysis

# Delete remote branch
git push origin --delete feature/new-analysis
```

## Git Flow Example

### 1. Start Release
```bash
git checkout develop
git pull origin develop
git checkout -b release/1.3.0
```

### 2. Prepare Release
```bash
# Update version
git commit -m "chore: bump version to 1.3.0"

# Final testing
git commit -m "test: final release testing"
```

### 3. Finish Release
```bash
git checkout main
git merge --no-ff release/1.3.0
git tag -a v1.3.0 -m "Release 1.3.0"

git checkout develop
git merge --no-ff release/1.3.0

# Delete release branch
git branch -d release/1.3.0
```

## Common Branch Patterns

### Long-Running Feature Branch
```bash
# Regular updates from develop
git checkout feature/long-running
git fetch origin
git rebase origin/develop

# Regular pushes
git push origin feature/long-running --force
```

### Experimental Branch
```bash
# Create from current state
git checkout -b experiment/new-algorithm

# Try things, may abandon
git add .
git commit -m "experiment: try new algorithm"

# If successful, merge to feature branch
git checkout feature/main-feature
git merge --no-ff experiment/new-algorithm

# If failed, just delete
git branch -D experiment/new-algorithm
```

### Patch Branch
```bash
# Apply patch from another branch
git checkout -b patch/security-fix
git cherry-pick abc123def456

# Test patch
git add .
git commit -m "test: verify security fix"

# Merge to appropriate branches
git checkout develop
git merge patch/security-fix
```