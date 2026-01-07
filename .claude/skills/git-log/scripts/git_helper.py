#!/usr/bin/env python3
"""
Git Helper Script for YouTube-SC Project

This script provides helper functions for common git operations.
Use as a reference for automating git workflows.

Note: This is an example script. Modify as needed for your specific use case.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple


class GitHelper:
    """Helper class for git operations."""

    def __init__(self, repo_path: str = "."):
        """
        Initialize git helper.

        Args:
            repo_path: Path to git repository (default: current directory)
        """
        self.repo_path = Path(repo_path).absolute()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def run_git_command(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """
        Run git command and return result.

        Args:
            args: Git command arguments
            check: Raise exception if command fails

        Returns:
            CompletedProcess object
        """
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            raise

    def status(self) -> str:
        """Get git status."""
        result = self.run_git_command(["status"])
        return result.stdout

    def add(self, files: List[str] = None) -> bool:
        """
        Stage files for commit.

        Args:
            files: List of files to stage (None = all files)

        Returns:
            True if successful
        """
        if files is None:
            result = self.run_git_command(["add", "."])
        else:
            result = self.run_git_command(["add"] + files)
        return result.returncode == 0

    def commit(self, message: str, allow_empty: bool = False) -> bool:
        """
        Commit staged changes.

        Args:
            message: Commit message
            allow_empty: Allow empty commit

        Returns:
            True if successful
        """
        cmd = ["commit", "-m", message]
        if allow_empty:
            cmd.append("--allow-empty")
        result = self.run_git_command(cmd)
        return result.returncode == 0

    def push(self, remote: str = "origin", branch: str = None) -> bool:
        """
        Push commits to remote repository.

        Args:
            remote: Remote name
            branch: Branch name (None = current branch)

        Returns:
            True if successful
        """
        if branch is None:
            # Get current branch
            result = self.run_git_command(["branch", "--show-current"])
            branch = result.stdout.strip()

        result = self.run_git_command(["push", remote, branch])
        return result.returncode == 0

    def pull(self, remote: str = "origin", branch: str = None) -> bool:
        """
        Pull latest changes from remote.

        Args:
            remote: Remote name
            branch: Branch name (None = current branch)

        Returns:
            True if successful
        """
        if branch is None:
            # Get current branch
            result = self.run_git_command(["branch", "--show-current"])
            branch = result.stdout.strip()

        result = self.run_git_command(["pull", remote, branch])
        return result.returncode == 0

    def create_branch(self, branch_name: str, switch: bool = True) -> bool:
        """
        Create new branch.

        Args:
            branch_name: Name of new branch
            switch: Switch to new branch after creation

        Returns:
            True if successful
        """
        cmd = ["checkout", "-b", branch_name] if switch else ["branch", branch_name]
        result = self.run_git_command(cmd)
        return result.returncode == 0

    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to existing branch.

        Args:
            branch_name: Branch to switch to

        Returns:
            True if successful
        """
        result = self.run_git_command(["checkout", branch_name])
        return result.returncode == 0

    def merge(self, branch_name: str, no_ff: bool = True) -> bool:
        """
        Merge branch into current branch.

        Args:
            branch_name: Branch to merge
            no_ff: Create merge commit even if fast-forward possible

        Returns:
            True if successful
        """
        cmd = ["merge", branch_name]
        if no_ff:
            cmd.append("--no-ff")
        result = self.run_git_command(cmd)
        return result.returncode == 0

    def log(self, limit: int = 10, oneline: bool = True) -> str:
        """
        Get commit history.

        Args:
            limit: Number of commits to show
            oneline: Show in one-line format

        Returns:
            Commit history
        """
        cmd = ["log", f"-{limit}"]
        if oneline:
            cmd.append("--oneline")
        result = self.run_git_command(cmd)
        return result.stdout

    def get_changed_files(self) -> List[str]:
        """
        Get list of changed files.

        Returns:
            List of changed file paths
        """
        result = self.run_git_command(["status", "--porcelain"])
        files = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Porcelain format: XY filename
                filename = line[3:].strip()
                files.append(filename)
        return files

    def has_changes(self) -> bool:
        """
        Check if there are uncommitted changes.

        Returns:
            True if there are changes
        """
        result = self.run_git_command(["status", "--porcelain"])
        return bool(result.stdout.strip())


def example_workflow():
    """Example workflow using GitHelper."""
    print("=== Git Helper Example Workflow ===")

    try:
        # Initialize helper
        helper = GitHelper(".")

        # Check status
        print("\n1. Current status:")
        print(helper.status())

        # Check for changes
        if helper.has_changes():
            print("\n2. Staging changes...")
            if helper.add():
                print("Changes staged successfully")

                print("\n3. Committing changes...")
                if helper.commit("feat: example commit from git helper"):
                    print("Commit successful")

                    print("\n4. Pushing to remote...")
                    if helper.push():
                        print("Push successful")
                    else:
                        print("Push failed")
                else:
                    print("Commit failed")
            else:
                print("Failed to stage changes")
        else:
            print("\nNo changes to commit")

        # Show recent commits
        print("\n5. Recent commits:")
        print(helper.log(limit=5))

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def create_feature_branch(branch_name: str):
    """Example: Create and work on feature branch."""
    print(f"\n=== Creating feature branch: {branch_name} ===")

    try:
        helper = GitHelper(".")

        # Create and switch to feature branch
        print(f"1. Creating branch '{branch_name}'...")
        if helper.create_branch(branch_name):
            print(f"Switched to branch '{branch_name}'")

            # Make some changes (example)
            print("\n2. Making changes...")
            # In real usage, you would modify files here

            print("\n3. Staging and committing...")
            if helper.add():
                if helper.commit(f"feat: start work on {branch_name}"):
                    print("Initial commit on feature branch")

                    print("\n4. Pushing feature branch...")
                    if helper.push(branch=branch_name):
                        print(f"Feature branch '{branch_name}' pushed to remote")
                    else:
                        print("Failed to push feature branch")
                else:
                    print("Failed to commit")
            else:
                print("Failed to stage changes")
        else:
            print("Failed to create branch")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Git Helper for YouTube-SC")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Example workflow command
    subparsers.add_parser("example", help="Run example workflow")

    # Create branch command
    branch_parser = subparsers.add_parser("branch", help="Create feature branch")
    branch_parser.add_argument("name", help="Branch name")

    # Status command
    subparsers.add_parser("status", help="Show git status")

    # Log command
    log_parser = subparsers.add_parser("log", help="Show git log")
    log_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of commits")

    args = parser.parse_args()

    if args.command == "example":
        example_workflow()
    elif args.command == "branch":
        create_feature_branch(args.name)
    elif args.command == "status":
        helper = GitHelper(".")
        print(helper.status())
    elif args.command == "log":
        helper = GitHelper(".")
        print(helper.log(limit=args.limit))
    else:
        parser.print_help()
        sys.exit(1)