# Management Tools

This project includes custom Claude Code skills for efficient project management, bug tracking, and environment management. These tools are integrated into the Claude Code CLI and provide specialized functionality for managing the YouTube-SC project.

## Available Skills

### 1. update-task-board
**Purpose**: Task management using markdown files

**Description**:
The `update-task-board` skill helps manage project tasks through a markdown-based task board located in `.claude/task-board.md`. It provides a structured way to track todo items, in-progress work, and completed tasks.

**Features**:
- Create and update task board with current project status
- Add new tasks with descriptions and priorities
- Mark tasks as in progress, completed, or blocked
- Generate progress reports and statistics
- Integrate with git to track task-related commits

**Usage**:
```bash
# Update task board with current status
claude /update-task-board

# Add a new task
claude /update-task-board --add "Implement feature X" --priority high

# Mark task as completed
claude /update-task-board --complete "Task description"

# Generate progress report
claude /update-task-board --report
```

**Task Board Structure**:
```markdown
# YouTube-SC Task Board

## To Do
- [ ] Task 1: Description
- [ ] Task 2: Description

## In Progress
- [x] Task 3: Description (started 2026-01-08)

## Completed (2026-01)
- [x] Task 4: Description (completed 2026-01-07)
- [x] Task 5: Description (completed 2026-01-06)
```

### 2. log-debug-issue
**Purpose**: Bug and issue tracking system

**Description**:
The `log-debug-issue` skill creates detailed bug reports and maintains a knowledge base of solutions in `.claude/bug-log.md`. It helps track, diagnose, and resolve issues encountered during development.

**Features**:
- Create structured bug reports with reproduction steps
- Log error messages and stack traces
- Track issue status (open, investigating, resolved)
- Document solutions for common problems
- Search existing issues before reporting new ones
- Generate issue statistics and trends

**Usage**:
```bash
# Log a new bug
claude /log-debug-issue --title "Import error in module X" --description "Detailed description"

# Add reproduction steps
claude /log-debug-issue --steps "1. Run command\n2. Observe error"

# Log error message
claude /log-debug-issue --error "ModuleNotFoundError: No module named 'missing_module'"

# Mark issue as resolved
claude /log-debug-issue --resolve --solution "Installed missing package"

# Search existing issues
claude /log-debug-issue --search "import error"
```

**Bug Log Structure**:
```markdown
# Bug Log

## Open Issues
### [2026-01-08] Import error in module X
**Status**: Investigating
**Description**: ModuleNotFoundError when running main.py
**Reproduction**:
1. Activate virtual environment
2. Run `python main.py`
3. Error occurs
**Error**: `ModuleNotFoundError: No module named 'missing_module'`
**Solution**:

## Resolved Issues
### [2026-01-07] Word2Vec download failure
**Status**: Resolved
**Solution**: Use TF-IDF features as fallback
**Resolution Date**: 2026-01-07
```

### 3. manage-python-env
**Purpose**: Python virtual environment management using uv

**Description**:
The `manage-python-env` skill helps create, maintain, and manage Python virtual environments using uv for fast and reliable environment setup. It ensures consistent development environments across different machines.

**Features**:
- Create new virtual environments with uv
- Install dependencies from requirements.txt
- Update packages and check for outdated dependencies
- Recreate environments from scratch
- Check environment health and package conflicts
- Generate environment reports

**Usage**:
```bash
# Create or update environment
claude /manage-python-env --setup

# Check environment health
claude /manage-python-env --check

# Update all packages
claude /manage-python-env --update

# Recreate environment from scratch
claude /manage-python-env --recreate

# Generate dependency report
claude /manage-python-env --report

# Install specific package
claude /manage-python-env --install package_name
```

**Environment Management**:
```bash
# Behind the scenes, the skill runs commands like:
uv venv .venv
uv pip install -r requirements.txt
uv pip list --outdated
uv update
```

## Integration with Claude Code

### Skill Location
Custom skills are located in `.claude/skills/` directory:
```
.claude/skills/
├── update-task-board/
│   ├── skill.json          # Skill configuration
│   ├── main.py             # Skill implementation
│   └── README.md           # Skill documentation
├── log-debug-issue/
│   ├── skill.json
│   ├── main.py
│   └── README.md
└── manage-python-env/
    ├── skill.json
    ├── main.py
    └── README.md
```

### Skill Configuration
Each skill has a `skill.json` configuration file:
```json
{
  "name": "update-task-board",
  "description": "Task management using markdown files",
  "version": "1.0.0",
  "author": "YouTube-SC Team",
  "commands": [
    {
      "name": "update",
      "description": "Update task board with current status"
    },
    {
      "name": "add",
      "description": "Add new task to board"
    }
  ]
}
```

## Usage Examples

### Project Setup Workflow
```bash
# 1. Clone repository
git clone https://github.com/yourusername/Youtube-SC.git
cd Youtube-SC

# 2. Setup environment using skill
claude /manage-python-env --setup

# 3. Check for existing issues
claude /log-debug-issue --search "setup"

# 4. Initialize task board
claude /update-task-board --init

# 5. Add initial tasks
claude /update-task-board --add "Explore codebase" --priority medium
claude /update-task-board --add "Run sentiment analysis" --priority high
```

### Daily Development Workflow
```bash
# Start work session
claude /update-task-board --start "Implement clustering visualization"

# Encounter bug
claude /log-debug-issue --title "Clustering visualization error" --error "ValueError: array shape mismatch"

# Work on fix
# ... coding ...

# Mark task as completed
claude /update-task-board --complete "Implement clustering visualization"

# Log solution
claude /log-debug-issue --resolve --solution "Fixed array shape calculation in visualizer.py"
```

### Team Collaboration
```bash
# Check team progress
claude /update-task-board --report --format team

# Review recent issues
claude /log-debug-issue --recent 5

# Update dependencies before team sync
claude /manage-python-env --update --dry-run

# Generate weekly status report
claude /update-task-board --weekly-report
claude /log-debug-issue --weekly-summary
```

## Advanced Features

### Automated Task Tracking
The skills can integrate with git to automatically track task completion:
```bash
# Link git commits to tasks
git commit -m "Add feature X

Closes: Implement feature X
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Skill automatically updates task board
claude /update-task-board --sync-git
```

### Custom Reports
Generate customized reports for different stakeholders:
```bash
# Technical report for developers
claude /update-task-board --report --format technical
claude /log-debug-issue --report --format technical

# Management report for supervisors
claude /update-task-board --report --format management
claude /log-debug-issue --report --format summary

# Export reports to files
claude /update-task-board --report --output progress_report.md
claude /log-debug-issue --report --output bug_report.md
```

### Integration with CI/CD
Skills can be used in automated pipelines:
```yaml
# GitHub Actions example
name: Project Management
on: [push, pull_request]

jobs:
  manage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
      - name: Update task board
        run: claude /update-task-board --ci
      - name: Check for critical issues
        run: claude /log-debug-issue --check-critical
```

## Customization

### Extending Skills
You can extend existing skills or create new ones:

1. **Add new commands**: Modify `skill.json` and `main.py`
2. **Custom templates**: Create custom markdown templates for reports
3. **Integration hooks**: Add pre/post execution hooks
4. **New skills**: Create new directory in `.claude/skills/`

### Configuration Options
Each skill supports configuration via environment variables or config files:
```bash
# Set task board location
export TASK_BOARD_PATH=".claude/custom-task-board.md"

# Set bug log format
export BUG_LOG_FORMAT="detailed"

# Set Python environment path
export PYTHON_ENV_PATH=".custom-venv"
```

## Best Practices

### Task Management
1. **Break down tasks**: Keep tasks small and actionable
2. **Regular updates**: Update task status daily
3. **Clear descriptions**: Write descriptive task titles
4. **Priority management**: Use priorities to focus work
5. **Archive completed**: Move old tasks to archive

### Issue Tracking
1. **Reproducible steps**: Always include reproduction steps
2. **Error details**: Capture complete error messages
3. **Solution documentation**: Document solutions thoroughly
4. **Regular review**: Review and close resolved issues weekly
5. **Prevention**: Update documentation to prevent recurring issues

### Environment Management
1. **Regular updates**: Update dependencies monthly
2. **Version locking**: Keep requirements.txt up to date
3. **Health checks**: Run environment checks weekly
4. **Backup configurations**: Backup environment configurations
5. **Document changes**: Document environment changes

## Troubleshooting Skills

### Common Issues
```bash
# Skill not found
# Ensure skills are in .claude/skills/ directory
ls .claude/skills/

# Permission errors
chmod +x .claude/skills/*/main.py

# Python path issues
# Ensure using project virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate
```

### Debug Mode
```bash
# Run skills with debug output
claude /update-task-board --debug
claude /log-debug-issue --verbose
claude /manage-python-env --debug
```

### Skill Logs
Check skill execution logs:
```bash
# Default log location
cat .claude/skill-logs/update-task-board.log
cat .claude/skill-logs/log-debug-issue.log
cat .claude/skill-logs/manage-python-env.log
```

## Future Enhancements

### Planned Features
1. **Git integration**: Automatic task tracking from git commits
2. **Time tracking**: Track time spent on tasks
3. **Team collaboration**: Multi-user task assignment and tracking
4. **Notification system**: Email/Slack notifications for updates
5. **Advanced analytics**: Predictive analytics for project completion

### Integration Possibilities
1. **JIRA/Asana integration**: Sync with external project management tools
2. **CI/CD integration**: Automated quality gates
3. **Documentation generation**: Auto-generate documentation from tasks
4. **Code review integration**: Link tasks to code reviews
5. **Performance monitoring**: Track performance metrics

These management tools provide a comprehensive system for managing the YouTube-SC project, from task tracking to environment management, all integrated within the Claude Code CLI for seamless development workflow.