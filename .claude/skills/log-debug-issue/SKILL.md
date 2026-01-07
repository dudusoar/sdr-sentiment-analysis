---
name: log-debug-issue
description: Bug and issue tracking system using markdown files. Record, track, and resolve bugs, errors, and issues encountered during development. Create detailed bug reports with reproduction steps, error messages, root cause analysis, and fixes. Use when debugging code, fixing errors, documenting problems, or maintaining a knowledge base of solutions.
---

# Debug Issue Logging Skill

This skill provides templates and workflows for systematically tracking bugs, errors, and their resolutions using markdown-based logs.

## Quick Start

1. **Create bug log**: Start a new bug log file for your project
2. **Report a bug**: Use the bug report template to document issues
3. **Investigate**: Add investigation notes and findings
4. **Resolve**: Document the fix and root cause
5. **Review**: Maintain a knowledge base of resolved issues

## Core Workflow

### When a Bug is Encountered

1. **Create bug report**:
   - Use the bug report template from `assets/bug-report-template.md`
   - Fill in all relevant sections
   - Include error messages, stack traces, and screenshots if available

2. **Investigate the issue**:
   - Document reproduction steps
   - Add investigation notes as you debug
   - Update status as you progress

3. **Resolve and document**:
   - Record the root cause
   - Document the fix
   - Add testing and verification steps
   - Update status to resolved

4. **Maintain knowledge base**:
   - Keep resolved bugs for future reference
   - Add to troubleshooting guides
   - Share solutions with team

## Bug Report Structure

A complete bug report should include:

- **Basic Info**: ID, title, status, priority, assignee
- **Description**: What happened vs what was expected
- **Reproduction**: Steps to reproduce the issue
- **Environment**: System, software versions, configuration
- **Investigation**: Debugging process and findings
- **Root Cause**: Analysis of why the issue occurred
- **Fix**: Solution implemented
- **Verification**: How the fix was tested
- **Prevention**: How to prevent similar issues

## Bug Severity Levels

### Critical
- System crashes, data loss, security vulnerabilities
- Requires immediate attention
- Blocks core functionality

### High
- Major functionality broken
- Workarounds available but cumbersome
- Affects multiple users

### Medium
- Minor functionality issues
- Cosmetic or UI problems
- Edge cases not working

### Low
- Typos, minor UI inconsistencies
- Enhancement suggestions
- Documentation issues

## Investigation Techniques

### Error Analysis
- Copy complete error messages and stack traces
- Note timestamps and frequency
- Identify patterns in when errors occur

### Reproduction
- Document exact steps to reproduce
- Note any preconditions or specific data needed
- Test on different environments if possible

### Debugging
- Add logging to trace execution flow
- Use debuggers to inspect state
- Check logs and monitoring tools

## Resolution Documentation

### Root Cause Analysis
- Explain why the bug occurred
- Identify underlying issues (not just symptoms)
- Consider design flaws, edge cases, or misunderstandings

### Fix Description
- Document code changes made
- Explain why this fix works
- Note any trade-offs or limitations

### Testing
- Describe how the fix was tested
- Include regression testing
- Document any new test cases added

## Integration with Development

### Version Control
- Link bug reports to commit hashes
- Reference pull request numbers
- Include code snippets in bug reports

### Project Management
- Sync bug status with task boards
- Update stakeholders on critical issues
- Use bug logs for sprint planning

### Team Collaboration
- Assign bugs to appropriate team members
- Add comments and updates
- Share findings in team meetings

## Resources

- **Bug Report Template**: See `assets/bug-report-template.md` for a complete template
- **Bug Examples**: See `references/bug-examples.md` for sample bug reports
- **Debugging Guide**: See `references/debugging-guide.md` for investigation techniques
- **Common Errors**: See `references/common-errors.md` for frequently encountered issues

## When to Use This Skill

Use this skill when:

- Encountering errors or bugs in code
- Debugging complex issues
- Documenting problems for team reference
- Building a knowledge base of solutions
- Analyzing recurring issues
- Preparing bug reports for issue trackers
- Learning from mistakes to prevent future issues