#!/usr/bin/env python3
"""
Test Runner Script for YouTube-SC Project

This script provides helper functions for running tests and generating reports.
Use as a reference for automating test workflows.

Note: This is an example script. Modify as needed for your specific use case.
"""

import subprocess
import sys
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class TestRunner:
    """Helper class for running tests."""

    def __init__(self, project_root: str = "."):
        """
        Initialize test runner.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root).absolute()
        self.tests_dir = self.project_root / "tests"

    def run_pytest(self, args: List[str] = None, module: str = None) -> subprocess.CompletedProcess:
        """
        Run pytest with given arguments.

        Args:
            args: Additional pytest arguments
            module: Specific module to test (e.g., "sdr_clustering_analysis")

        Returns:
            CompletedProcess object
        """
        cmd = ["pytest"]

        if module:
            # Test specific module
            module_test_dir = self.tests_dir / module
            if module_test_dir.exists():
                cmd.append(str(module_test_dir))
            else:
                print(f"Warning: Test directory not found: {module_test_dir}")

        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False  # Don't raise on test failures
            )
            return result
        except Exception as e:
            print(f"Failed to run pytest: {e}")
            raise

    def run_all_tests(self, verbose: bool = False) -> bool:
        """
        Run all tests.

        Args:
            verbose: Show verbose output

        Returns:
            True if all tests passed
        """
        args = ["-v"] if verbose else []
        result = self.run_pytest(args)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    def run_module_tests(self, module: str, verbose: bool = False) -> bool:
        """
        Run tests for specific module.

        Args:
            module: Module name (e.g., "sdr_clustering_analysis")
            verbose: Show verbose output

        Returns:
            True if tests passed
        """
        args = ["-v"] if verbose else []
        result = self.run_pytest(args, module=module)

        print(f"\n=== {module.upper()} Tests ===")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    def run_with_coverage(self, module: str = None) -> Dict:
        """
        Run tests with coverage report.

        Args:
            module: Specific module to measure coverage for

        Returns:
            Coverage statistics
        """
        cov_args = ["--cov=.", "--cov-report=term", "--cov-report=html"]

        if module:
            cov_args[0] = f"--cov=Code/{module}"

        result = self.run_pytest(cov_args, module=module)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Parse coverage from output (simplified)
        coverage = self._parse_coverage_output(result.stdout)
        return coverage

    def _parse_coverage_output(self, output: str) -> Dict:
        """
        Parse coverage statistics from pytest output.

        Args:
            output: pytest coverage output

        Returns:
            Dictionary with coverage stats
        """
        coverage = {
            "total": 0.0,
            "modules": {}
        }

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if "TOTAL" in line and "%" in line:
                # Extract total percentage
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        try:
                            coverage["total"] = float(part.replace("%", ""))
                        except ValueError:
                            pass
            elif ".py" in line and "%" in line:
                # Parse module coverage line
                parts = line.split()
                if len(parts) >= 2:
                    module_name = parts[0]
                    for part in parts[1:]:
                        if "%" in part:
                            try:
                                module_cov = float(part.replace("%", ""))
                                coverage["modules"][module_name] = module_cov
                            except ValueError:
                                pass

        return coverage

    def run_marked_tests(self, marker: str, verbose: bool = False) -> bool:
        """
        Run tests with specific marker.

        Args:
            marker: pytest marker (e.g., "clustering", "slow", "integration")
            verbose: Show verbose output

        Returns:
            True if tests passed
        """
        args = ["-v", "-m", marker] if verbose else ["-m", marker]
        result = self.run_pytest(args)

        print(f"\n=== Tests with marker '{marker}' ===")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    def generate_html_report(self, output_dir: str = "test_reports") -> str:
        """
        Generate HTML test report.

        Args:
            output_dir: Output directory for reports

        Returns:
            Path to HTML report
        """
        report_dir = self.project_root / output_dir
        report_dir.mkdir(exist_ok=True)

        html_file = report_dir / "report.html"
        args = [
            "--html", str(html_file),
            "--self-contained-html"
        ]

        result = self.run_pytest(args)
        print(f"HTML report generated: {html_file}")

        if result.returncode != 0:
            print("Some tests failed, but report was generated")

        return str(html_file)

    def generate_junit_report(self, output_dir: str = "test_reports") -> str:
        """
        Generate JUnit XML test report.

        Args:
            output_dir: Output directory for reports

        Returns:
            Path to XML report
        """
        report_dir = self.project_root / output_dir
        report_dir.mkdir(exist_ok=True)

        xml_file = report_dir / "junit.xml"
        args = ["--junit-xml", str(xml_file)]

        result = self.run_pytest(args)
        print(f"JUnit XML report generated: {xml_file}")

        # Parse and display summary
        if xml_file.exists():
            summary = self._parse_junit_report(xml_file)
            print(f"Tests: {summary['total']}, Failures: {summary['failures']}, Errors: {summary['errors']}")

        return str(xml_file)

    def _parse_junit_report(self, xml_path: Path) -> Dict:
        """
        Parse JUnit XML report.

        Args:
            xml_path: Path to JUnit XML file

        Returns:
            Test summary dictionary
        """
        summary = {
            "total": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "time": 0.0
        }

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for testsuite in root.findall("testsuite"):
                summary["total"] += int(testsuite.get("tests", 0))
                summary["failures"] += int(testsuite.get("failures", 0))
                summary["errors"] += int(testsuite.get("errors", 0))
                summary["skipped"] += int(testsuite.get("skipped", 0))
                summary["time"] += float(testsuite.get("time", 0))

        except Exception as e:
            print(f"Error parsing JUnit report: {e}")

        return summary

    def list_test_modules(self) -> List[str]:
        """
        List available test modules.

        Returns:
            List of module names with tests
        """
        modules = []
        if self.tests_dir.exists():
            for item in self.tests_dir.iterdir():
                if item.is_dir() and not item.name.startswith("__"):
                    modules.append(item.name)
        return sorted(modules)

    def list_test_markers(self) -> List[str]:
        """
        List available test markers from pytest.ini.

        Returns:
            List of marker names
        """
        markers = []
        pytest_ini = self.project_root / "pytest.ini"
        if pytest_ini.exists():
            try:
                with open(pytest_ini, 'r') as f:
                    in_markers_section = False
                    for line in f:
                        line = line.strip()
                        if line == "[pytest]":
                            continue
                        if line == "markers =":
                            in_markers_section = True
                            continue
                        if in_markers_section:
                            if line.startswith("["):
                                break  # Next section
                            if ":" in line:
                                marker_name = line.split(":")[0].strip()
                                markers.append(marker_name)
            except Exception as e:
                print(f"Error reading pytest.ini: {e}")

        return markers


def example_workflow():
    """Example test workflow."""
    print("=== Test Runner Example Workflow ===")

    try:
        runner = TestRunner(".")

        # List available modules
        print("\n1. Available test modules:")
        modules = runner.list_test_modules()
        for module in modules:
            print(f"  - {module}")

        # Run tests for each module
        print("\n2. Running module tests...")
        for module in modules:
            print(f"\n--- Testing {module} ---")
            success = runner.run_module_tests(module, verbose=True)
            if success:
                print(f"✓ {module} tests passed")
            else:
                print(f"✗ {module} tests failed")

        # Generate coverage report
        print("\n3. Generating coverage report...")
        coverage = runner.run_with_coverage()
        print(f"\nTotal coverage: {coverage.get('total', 0):.1f}%")

        # Generate HTML report
        print("\n4. Generating HTML report...")
        html_report = runner.generate_html_report()
        print(f"Report: {html_report}")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def check_coverage_threshold(threshold: float = 80.0):
    """Check if test coverage meets threshold."""
    print(f"\n=== Checking coverage threshold ({threshold}%) ===")

    try:
        runner = TestRunner(".")
        coverage = runner.run_with_coverage()

        total_coverage = coverage.get("total", 0.0)
        print(f"\nTotal coverage: {total_coverage:.1f}%")

        if total_coverage >= threshold:
            print(f"✓ Coverage meets threshold of {threshold}%")
            return True
        else:
            print(f"✗ Coverage below threshold of {threshold}%")
            print("\nModules below threshold:")
            for module, cov in coverage.get("modules", {}).items():
                if cov < threshold:
                    print(f"  - {module}: {cov:.1f}%")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Runner for YouTube-SC")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Example workflow
    subparsers.add_parser("example", help="Run example workflow")

    # Run all tests
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Run module tests
    module_parser = subparsers.add_parser("module", help="Run tests for specific module")
    module_parser.add_argument("name", help="Module name")
    module_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Run marked tests
    marker_parser = subparsers.add_parser("marker", help="Run tests with specific marker")
    marker_parser.add_argument("name", help="Marker name")
    marker_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Run tests with coverage")
    coverage_parser.add_argument("--module", help="Specific module for coverage")
    coverage_parser.add_argument("--threshold", type=float, help="Minimum coverage threshold")

    # List commands
    subparsers.add_parser("list-modules", help="List available test modules")
    subparsers.add_parser("list-markers", help="List available test markers")

    # Generate reports
    report_parser = subparsers.add_parser("report", help="Generate test reports")
    report_parser.add_argument("--html", action="store_true", help="Generate HTML report")
    report_parser.add_argument("--junit", action="store_true", help="Generate JUnit XML report")

    args = parser.parse_args()

    if args.command == "example":
        example_workflow()
    elif args.command == "all":
        runner = TestRunner(".")
        success = runner.run_all_tests(verbose=args.verbose)
        sys.exit(0 if success else 1)
    elif args.command == "module":
        runner = TestRunner(".")
        success = runner.run_module_tests(args.name, verbose=args.verbose)
        sys.exit(0 if success else 1)
    elif args.command == "marker":
        runner = TestRunner(".")
        success = runner.run_marked_tests(args.name, verbose=args.verbose)
        sys.exit(0 if success else 1)
    elif args.command == "coverage":
        runner = TestRunner(".")
        coverage = runner.run_with_coverage(module=args.module)
        if args.threshold:
            meets_threshold = coverage.get("total", 0) >= args.threshold
            sys.exit(0 if meets_threshold else 1)
    elif args.command == "list-modules":
        runner = TestRunner(".")
        modules = runner.list_test_modules()
        print("Available test modules:")
        for module in modules:
            print(f"  - {module}")
    elif args.command == "list-markers":
        runner = TestRunner(".")
        markers = runner.list_test_markers()
        print("Available test markers:")
        for marker in markers:
            print(f"  - {marker}")
    elif args.command == "report":
        runner = TestRunner(".")
        if args.html:
            runner.generate_html_report()
        if args.junit:
            runner.generate_junit_report()
        if not (args.html or args.junit):
            print("Please specify --html and/or --junit")
    else:
        parser.print_help()
        sys.exit(1)