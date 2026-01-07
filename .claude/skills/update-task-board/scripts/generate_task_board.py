#!/usr/bin/env python3
"""
Generate a task board markdown file from a template.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

def create_task_board(project_name, output_path=None):
    """
    Create a new task board markdown file.
    
    Args:
        project_name: Name of the project
        output_path: Path to save the task board (default: task-board.md in current directory)
    """
    if output_path is None:
        output_path = "task-board.md"
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Read template
    template_path = Path(__file__).parent.parent / "assets" / "task-board-template.md"
    
    if not template_path.exists():
        # Create basic template if not found
        template = """# Project Task Board

## Project Overview
- **Project Name**: {project_name}
- **Objective**: [Brief project objective]
- **Start Date**: {current_date}
- **Target Completion**: [YYYY-MM-DD]
- **Status**: `planning`
- **Last Updated**: {current_date}

## Progress Summary
- **Total Tasks**: 0
- **Completed**: 0 (0%)
- **In Progress**: 0 (0%)
- **Pending**: 0 (0%)

## Task List

[Add tasks here using the task template]

## Change Log

| Date | Task | Change | Notes |
|------|------|--------|-------|
| {current_date} | Initial board | Created | Task board created |

## Key Decisions & Notes

[Document important decisions here]

## Next Steps

1. Define project objectives
2. Break down into tasks
3. Assign priorities and owners

## Blockers & Risks

| Blocker/Risk | Impact | Mitigation | Owner |
|--------------|--------|------------|-------|
| [Add blockers or risks here] | [High/Medium/Low] | [Action plan] | [Person] |
"""
    else:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    
    # Replace placeholders
    content = template.replace("{project_name}", project_name)
    content = content.replace("{current_date}", current_date)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Task board created: {output_path}")
    print(f"   Project: {project_name}")
    print(f"   Date: {current_date}")
    
    return output_path

def add_task_to_board(task_title, task_description, board_path="task-board.md"):
    """
    Add a new task to an existing task board.
    
    Args:
        task_title: Title of the task
        task_description: Description of the task
        board_path: Path to the task board file
    """
    if not os.path.exists(board_path):
        print(f"❌ Task board not found: {board_path}")
        return False
    
    with open(board_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Task List section
    if "## Task List" not in content:
        print("❌ Task List section not found in task board")
        return False
    
    # Count existing tasks to get next task number
    task_count = content.count("### ")  # Count existing task headings
    
    # Create new task
    new_task = f"""
### {task_count + 1}. {task_title}

**Status**: `pending`
**Priority**: `medium`
**Assignee**: [Person/Team]
**Due Date**: [YYYY-MM-DD]

**Description**:
{task_description}

**Subtasks**:
- [ ] Define subtasks

**Notes**:
[Additional context or dependencies]

**Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    
    # Insert new task after the Task List section
    sections = content.split("## Task List")
    if len(sections) < 2:
        print("❌ Could not parse task board structure")
        return False
    
    # Reconstruct content with new task
    new_content = sections[0] + "## Task List" + sections[1].replace("\n\n", "\n\n" + new_task + "\n\n", 1)
    
    # Update task count in Progress Summary
    import re
    total_match = re.search(r"Total Tasks\": (\d+)", new_content)
    if total_match:
        current_total = int(total_match.group(1))
        new_content = re.sub(r"Total Tasks\": \d+", f"Total Tasks\": {current_total + 1}", new_content)
    
    # Write updated content
    with open(board_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Task added: {task_title}")
    print(f"   Task #{task_count + 1} added to {board_path}")
    
    return True

def update_task_status(board_path, task_number, new_status):
    """
    Update the status of a task.
    
    Args:
        board_path: Path to the task board file
        task_number: Task number to update (1-based)
        new_status: New status (pending, in_progress, completed)
    """
    if not os.path.exists(board_path):
        print(f"❌ Task board not found: {board_path}")
        return False
    
    valid_statuses = ["pending", "in_progress", "completed", "blocked"]
    if new_status not in valid_statuses:
        print(f"❌ Invalid status: {new_status}. Must be one of {valid_statuses}")
        return False
    
    with open(board_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the task
    task_pattern = f"### {task_number}\\. .*?(?=### {task_number + 1}\\. |## |$)"
    match = re.search(task_pattern, content, re.DOTALL)
    
    if not match:
        print(f"❌ Task {task_number} not found")
        return False
    
    task_content = match.group(0)
    
    # Update status
    updated_task = re.sub(
        r"Status\": `[^`]+`",
        f"Status\": `{new_status}`",
        task_content
    )
    
    # Update last updated timestamp
    updated_task = re.sub(
        r"Last Updated\": [^\n]+",
        f"Last Updated\": {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        updated_task
    )
    
    # Replace task in content
    new_content = content.replace(task_content, updated_task)
    
    # Write updated content
    with open(board_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Task {task_number} status updated to: {new_status}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Manage task boards")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new task board")
    create_parser.add_argument("project_name", help="Name of the project")
    create_parser.add_argument("--output", "-o", help="Output file path")
    
    # Add task command
    add_parser = subparsers.add_parser("add", help="Add a task to a task board")
    add_parser.add_argument("title", help="Task title")
    add_parser.add_argument("description", help="Task description")
    add_parser.add_argument("--board", "-b", default="task-board.md", help="Task board file path")
    
    # Update status command
    status_parser = subparsers.add_parser("status", help="Update task status")
    status_parser.add_argument("task_number", type=int, help="Task number to update")
    status_parser.add_argument("new_status", help="New status (pending, in_progress, completed, blocked)")
    status_parser.add_argument("--board", "-b", default="task-board.md", help="Task board file path")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_task_board(args.project_name, args.output)
    elif args.command == "add":
        add_task_to_board(args.title, args.description, args.board)
    elif args.command == "status":
        update_task_status(args.board, args.task_number, args.new_status)
    else:
        parser.print_help()

if __name__ == "__main__":
    import re  # Import here for the update_task_status function
    main()