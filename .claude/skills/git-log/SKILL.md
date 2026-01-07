---
name: git-log
description: Git version control management for code commits, push operations, branch management, and maintaining git history records. Use when managing git operations in software projects, including committing code changes, pushing to remote repositories, viewing git logs, creating branches, and resolving merge conflicts.
---

# Git Version Control Management Skill

This skill provides tools and workflows for managing git operations in software development projects. It supports code commits, push operations, branch management, and git history maintenance.

## Quick Start

1. **Check git status**: Review current changes before committing
2. **Stage changes**: Add modified files to staging area
3. **Commit changes**: Create commits with descriptive messages
4. **Push to remote**: Upload commits to remote repository
5. **Review history**: View git log to understand project history

## Core Workflow

### Committing Code Changes

When making code changes:

1. Check current status: `git status`
2. Stage files: `git add <file>` or `git add .`
3. Review staged changes: `git diff --cached`
4. Commit with message: `git commit -m "Descriptive message"`
5. Push to remote: `git push origin <branch>`

### Branch Management

For feature development:

1. Create new branch: `git checkout -b feature/new-feature`
2. Work on branch: Make commits on the feature branch
3. Switch branches: `git checkout main` or `git checkout feature/new-feature`
4. Merge branches: `git merge feature/new-feature` (when ready)
5. Delete old branches: `git branch -d feature/old-feature`

### Git History Review

To understand project history:

1. View commit history: `git log --oneline --graph --all`
2. Check specific commit: `git show <commit-hash>`
3. Compare branches: `git diff main..feature/new-feature`
4. View file history: `git log --follow -p <file-path>`

### Remote Repository Operations

For collaboration:

1. Check remote: `git remote -v`
2. Fetch updates: `git fetch origin`
3. Pull changes: `git pull origin main`
4. Push changes: `git push origin <branch>`
5. Set upstream: `git push -u origin <branch>` (first push)

## Common Git Patterns

### Standard Commit Workflow
```bash
# 1. Check status
git status

# 2. Stage all changes
git add .

# 3. Commit with message
git commit -m "feat: add new feature"

# 4. Push to remote
git push origin main
```

### Feature Branch Workflow
```bash
# 1. Create and switch to feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "feat: implement new feature"

# 3. Push feature branch
git push -u origin feature/new-feature

# 4. Merge to main (after review)
git checkout main
git pull origin main
git merge feature/new-feature
git push origin main
```

### Git Log Analysis
```bash
# Compact view with graph
git log --oneline --graph --all

# Show recent commits with details
git log -5 --stat

# Search commits by message
git log --grep="bugfix"

# View commits by date
git log --since="2024-01-01" --until="2024-12-31"
```

## Error Handling

### Common Issues and Solutions

#### 1. Commit Rejected (Non-Fast-Forward)
```bash
# Pull latest changes before pushing
git pull origin main --rebase

# Or force push (use with caution)
git push origin main --force
```

#### 2. Merge Conflicts
```bash
# Resolve conflicts in files marked by <<<<<<<
# After resolving:
git add .
git commit -m "fix: resolve merge conflicts"
```

#### 3. Accidental Commit
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

#### 4. Wrong Branch Commits
```bash
# Stash changes
git stash

# Switch to correct branch
git checkout correct-branch

# Apply stashed changes
git stash pop
```

## Best Practices

### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Keep descriptions concise but descriptive
- Reference issues/tickets when applicable

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Git Hygiene
- Commit frequently with logical changes
- Keep commits focused on single purpose
- Write descriptive commit messages
- Review changes before committing
- Keep repository clean and organized

## Resources

- **Git Cheat Sheet**: See `references/git-cheatsheet.md` for common commands
- **Commit Examples**: See `references/commit-examples.md` for good commit messages
- **Branch Examples**: See `references/branch-examples.md` for branch management patterns
- **YouTube-SC Examples**: See `examples/youtube-sc/` for project-specific git workflows

## When to Use This Skill

Use this skill when:
- Committing code changes to version control
- Managing git branches for feature development
- Pushing code to remote repositories
- Reviewing git history and understanding changes
- Resolving git conflicts and issues
- Maintaining clean git history and project structure