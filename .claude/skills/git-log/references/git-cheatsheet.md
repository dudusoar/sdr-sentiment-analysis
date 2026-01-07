# Git Command Cheat Sheet

## Basic Commands

### Repository Operations
```bash
# Initialize new repository
git init

# Clone existing repository
git clone <repository-url>

# Check repository status
git status

# View remote repositories
git remote -v

# Add remote repository
git remote add origin <url>
```

### Staging and Committing
```bash
# Stage specific file
git add <filename>

# Stage all changes
git add .

# Stage with patch mode (interactive)
git add -p

# Commit staged changes
git commit -m "Commit message"

# Commit with detailed message editor
git commit

# Amend last commit
git commit --amend

# Unstage file
git reset <filename>

# Discard changes in working directory
git checkout -- <filename>
```

### Branch Operations
```bash
# List branches
git branch

# List all branches (including remote)
git branch -a

# Create new branch
git branch <branch-name>

# Switch to branch
git checkout <branch-name>

# Create and switch to new branch
git checkout -b <branch-name>

# Delete branch
git branch -d <branch-name>

# Force delete branch
git branch -D <branch-name>

# Rename branch
git branch -m <old-name> <new-name>
```

### Merging and Rebasing
```bash
# Merge branch into current branch
git merge <branch-name>

# Abort merge in progress
git merge --abort

# Rebase current branch onto another
git rebase <branch-name>

# Interactive rebase (last 3 commits)
git rebase -i HEAD~3

# Abort rebase
git rebase --abort

# Continue rebase after resolving conflicts
git rebase --continue
```

## History and Logging

### Viewing History
```bash
# Show commit history
git log

# One-line format
git log --oneline

# Graph format
git log --graph

# Show all branches
git log --all

# Combined formats
git log --oneline --graph --all

# Show changes in commits
git log -p

# Show statistics
git log --stat

# Search commits by message
git log --grep="search term"

# Filter by date
git log --since="2024-01-01" --until="2024-12-31"

# Filter by author
git log --author="name"

# Show specific number of commits
git log -10
```

### Comparing Changes
```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --cached

# Compare working directory with specific commit
git diff <commit-hash>

# Compare two commits
git diff <commit1> <commit2>

# Compare branches
git diff main..feature-branch

# Show only file names that changed
git diff --name-only
```

### Examining Commits
```bash
# Show specific commit
git show <commit-hash>

# Show commit and surrounding changes
git show <commit-hash> -p

# Show commit statistics
git show <commit-hash> --stat
```

## Remote Operations

### Fetching and Pulling
```bash
# Fetch from remote
git fetch origin

# Fetch specific branch
git fetch origin <branch-name>

# Pull changes (fetch + merge)
git pull origin <branch-name>

# Pull with rebase
git pull --rebase origin <branch-name>
```

### Pushing
```bash
# Push to remote
git push origin <branch-name>

# Push and set upstream
git push -u origin <branch-name>

# Force push (use with caution)
git push --force origin <branch-name>

# Push tags
git push --tags
```

### Tags
```bash
# List tags
git tag

# Create lightweight tag
git tag <tag-name>

# Create annotated tag
git tag -a <tag-name> -m "Tag message"

# Push tags to remote
git push origin --tags

# Delete tag
git tag -d <tag-name>
```

## Undoing Changes

### Reset Operations
```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes unstaged)
git reset --mixed HEAD~1

# Hard reset (discard all changes)
git reset --hard HEAD~1

# Reset specific file
git reset HEAD <filename>

# Reset to specific commit
git reset --hard <commit-hash>
```

### Revert Operations
```bash
# Revert specific commit
git revert <commit-hash>

# Revert without committing
git revert -n <commit-hash>

# Revert merge commit
git revert -m 1 <merge-commit-hash>
```

### Stashing
```bash
# Stash changes
git stash

# Stash with message
git stash save "Stash message"

# List stashes
git stash list

# Apply stash
git stash apply

# Apply specific stash
git stash apply stash@{0}

# Pop stash (apply and remove)
git stash pop

# Drop stash
git stash drop stash@{0}

# Clear all stashes
git stash clear
```

## Configuration

### User Configuration
```bash
# Set global username
git config --global user.name "Your Name"

# Set global email
git config --global user.email "email@example.com"

# Set editor
git config --global core.editor "code --wait"

# List all configurations
git config --list

# Get specific configuration
git config user.name
```

### Alias Configuration
```bash
# Create aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.lg "log --oneline --graph --all"

# Use aliases
git co main
git br feature/new-feature
git ci -m "Message"
git st
git lg
```

## Advanced Operations

### Bisect
```bash
# Start bisect session
git bisect start

# Mark commit as bad
git bisect bad

# Mark commit as good
git bisect good <good-commit>

# Automatically bisect
git bisect run <test-script>

# End bisect session
git bisect reset
```

### Submodules
```bash
# Add submodule
git submodule add <repository-url>

# Initialize submodules
git submodule init

# Update submodules
git submodule update

# Clone with submodules
git clone --recursive <repository-url>
```

### Worktree
```bash
# Create new worktree
git worktree add <path> <branch>

# List worktrees
git worktree list

# Remove worktree
git worktree remove <path>
```

## Useful Combinations

### Clean Up Local Branches
```bash
# Delete merged branches
git branch --merged | grep -v "\*" | xargs -n 1 git branch -d

# Delete remote-tracking branches for deleted remotes
git remote prune origin
```

### Update Fork
```bash
# Add upstream remote
git remote add upstream <original-repo-url>

# Fetch upstream changes
git fetch upstream

# Merge upstream changes
git merge upstream/main

# Push to your fork
git push origin main
```

### Search in Code
```bash
# Search for string in code
git grep "search string"

# Search with line numbers
git grep -n "search string"

# Search across all branches
git grep "search string" $(git rev-list --all)
```

## Common Workflows

### Feature Development
```bash
# Start new feature
git checkout -b feature/new-feature
# Make changes and commit
git add .
git commit -m "feat: implement new feature"
# Push feature branch
git push -u origin feature/new-feature
# Create pull request, then merge
```

### Hotfix Development
```bash
# Create hotfix branch from main
git checkout -b hotfix/issue-123 main
# Fix the issue and commit
git add .
git commit -m "fix: resolve issue #123"
# Merge to main and develop
git checkout main
git merge hotfix/issue-123
git checkout develop
git merge hotfix/issue-123
# Clean up
git branch -d hotfix/issue-123
```

### Release Preparation
```bash
# Create release branch
git checkout -b release/1.0.0 develop
# Bump version, update changelog
git add .
git commit -m "chore: prepare release 1.0.0"
# Merge to main and tag
git checkout main
git merge release/1.0.0
git tag -a v1.0.0 -m "Release 1.0.0"
# Merge back to develop
git checkout develop
git merge release/1.0.0
# Clean up
git branch -d release/1.0.0
```