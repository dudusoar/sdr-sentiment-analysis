# Task Examples

## Software Development Tasks

### Code Refactoring Task
```markdown
### 1. Refactor authentication module

**Status**: `in_progress`
**Priority**: `high`
**Assignee**: Development Team
**Due Date**: 2024-01-15

**Description**:
Refactor the legacy authentication module to use OAuth 2.0 and JWT tokens. Current implementation uses session-based authentication which has security vulnerabilities and scalability issues.

**Subtasks**:
- [x] Research OAuth 2.0 implementation patterns
- [x] Design new authentication flow
- [ ] Implement OAuth 2.0 provider integration
- [ ] Add JWT token generation and validation
- [ ] Update API endpoints to use new authentication
- [ ] Write unit tests for new authentication module
- [ ] Update documentation

**Notes**:
- Need to maintain backward compatibility during transition
- Security audit required before deployment
- Coordinate with frontend team for UI updates

**Last Updated**: 2024-01-10 14:30
```

### Bug Fix Task
```markdown
### 2. Fix memory leak in data processing pipeline

**Status**: `completed`
**Priority**: `critical`
**Assignee**: Senior Developer
**Due Date**: 2024-01-05

**Description**:
Investigate and fix memory leak causing server crashes during large data processing jobs. Issue occurs when processing files >100MB.

**Subtasks**:
- [x] Reproduce the memory leak
- [x] Profile memory usage to identify leak source
- [x] Fix unclosed database connections
- [x] Implement proper resource cleanup
- [x] Test with large datasets
- [x] Deploy fix to staging

**Notes**:
- Root cause: Database connection pool not being properly released
- Fix: Added context managers for all database operations
- Testing confirmed memory usage stable at 500MB for 1GB file

**Last Updated**: 2024-01-05 11:45
```

## Project Management Tasks

### Planning Task
```markdown
### 3. Create project implementation plan

**Status**: `pending`
**Priority**: `medium`
**Assignee**: Project Manager
**Due Date**: 2024-01-20

**Description**:
Develop detailed implementation plan for Q2 feature rollout, including timeline, resource allocation, and risk assessment.

**Subtasks**:
- [ ] Gather requirements from stakeholders
- [ ] Break down features into development tasks
- [ ] Estimate effort for each task
- [ ] Create timeline with milestones
- [ ] Identify resource requirements
- [ ] Document risks and mitigation strategies
- [ ] Present plan to leadership

**Notes**:
- Need input from engineering, design, and product teams
- Consider dependencies with other projects
- Include buffer time for unexpected delays

**Last Updated**: 2024-01-08 09:15
```

## Code Review Tasks

### Code Review Task
```markdown
### 4. Review pull request #123: User profile enhancements

**Status**: `in_progress`
**Priority**: `medium`
**Assignee**: Senior Developer
**Due Date**: 2024-01-12

**Description**:
Review and provide feedback on PR #123 implementing user profile enhancements including avatar upload, profile editing, and privacy settings.

**Subtasks**:
- [x] Review code for security issues
- [ ] Check for performance implications
- [ ] Verify edge cases are handled
- [ ] Test functionality locally
- [ ] Provide constructive feedback
- [ ] Approve or request changes

**Notes**:
- Focus on security of file upload functionality
- Ensure privacy settings work correctly
- Check mobile responsiveness

**Last Updated**: 2024-01-11 16:20
```

## Template Variations

### Simple Task (Minimal)
```markdown
### [Number]. [Brief Title]

**Status**: `pending`
**Description**: [One-sentence description]
```

### Detailed Task (Comprehensive)
```markdown
### [Number]. [Task Title]

**Status**: [status]
**Priority**: [priority]
**Assignee**: [person/team]
**Due Date**: YYYY-MM-DD
**Estimate**: [hours/days]
**Tags**: [tag1, tag2, tag3]

**Description**:
[Detailed description]

**Acceptance Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

**Subtasks**:
- [ ] Subtask 1
- [ ] Subtask 2

**Dependencies**:
- Depends on: [Task #]
- Blocks: [Task #]

**Notes**:
[Additional context]

**Testing Instructions**:
[Steps to verify completion]

**Last Updated**: YYYY-MM-DD HH:MM
```

## Status Transition Examples

### Starting Work
```markdown
**Status Change**: `pending` → `in_progress`
**Note**: Began implementing feature. Estimated 3 days completion.
```

### Completing Work
```markdown
**Status Change**: `in_progress` → `completed`
**Note**: Feature implemented and tested. All acceptance criteria met.
```

### Blocked Task
```markdown
**Status Change**: `in_progress` → `blocked`
**Note**: Waiting on API documentation from third-party vendor. Estimated 2-day delay.
```