---
name: update-task-board
description: Task management and tracking system using markdown files. Create and maintain task boards to track project progress, status, and priorities. Use when starting new projects, refactoring code, or managing complex multi-step tasks that require organized tracking.
---

# Task Board Management Skill

This skill provides tools and templates for managing project tasks using markdown-based task boards.

## Quick Start

1. **Create a new task board**: Use the task board template to start tracking your project
2. **Add tasks**: Use the task template to add individual tasks with details
3. **Update progress**: Mark tasks as pending, in_progress, or completed
4. **Review status**: Generate status reports from the task board

## Core Workflow

### Creating a Task Board

When starting a new project or refactoring:

1. Create a `task-board.md` file in your project root
2. Copy the task board template from `assets/task-board-template.md`
3. Customize the project name, objectives, and initial tasks
4. Use the task template for adding detailed tasks

### Adding and Managing Tasks

For each task:

1. Use the task template format
2. Include essential information:
   - Task description (content)
   - Status (pending/in_progress/completed)
   - Active form description
   - Optional: priority, assignee, due date
3. Update status as work progresses

### Generating Status Reports

To get an overview of project progress:

1. Review the task board markdown file
2. Count tasks by status
3. Identify blockers or overdue tasks
4. Update stakeholders with current status

## Task Board Structure

A task board should include:

- **Project overview**: Name, objectives, timeline
- **Task list**: All tasks with status tracking
- **Progress metrics**: Completed vs total tasks
- **Recent updates**: Change log of task status changes

## Task Template

```markdown
### [Task Number]. [Task Title]

**Status**: `pending` | `in_progress` | `completed`
**Priority**: `high` | `medium` | `low` (optional)
**Assignee**: [Person/Team] (optional)
**Due Date**: YYYY-MM-DD (optional)

**Description**:
[Detailed task description]

**Subtasks**:
- [ ] Subtask 1
- [ ] Subtask 2

**Notes**:
[Additional context or dependencies]

**Last Updated**: YYYY-MM-DD HH:MM
```

## Integration with Claude Code

When working with Claude Code:

1. Use the TodoWrite tool to track real-time task status
2. Sync the task board with the current todo list
3. Update task status when completing work items
4. Document decisions and changes in task notes

## Best Practices

### Task Definition
- Create specific, actionable tasks
- Break complex tasks into smaller subtasks
- Include acceptance criteria when possible
- Assign realistic priorities and due dates

### Status Updates
- Update task status immediately when work begins/completes
- Add notes explaining status changes
- Regularly review and prioritize tasks
- Archive completed tasks periodically

### Collaboration
- Use clear task descriptions for team understanding
- Include relevant context and dependencies
- Document decisions and rationale in task notes
- Share task board updates with stakeholders

## Resources

- **Task Board Template**: See `assets/task-board-template.md` for a complete starting template
- **Task Examples**: See `references/task-examples.md` for sample tasks in different contexts
- **Progress Reporting**: See `references/progress-reporting.md` for status report templates

## When to Use This Skill

Use this skill when:

- Starting a new software project or refactoring
- Managing complex multi-step tasks
- Tracking progress for team projects
- Documenting work for reports or presentations
- Organizing work into manageable chunks