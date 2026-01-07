# Bug Report Examples

## Example 1: Critical Database Connection Issue

```markdown
# Bug Report: DB-2024-001

## Basic Information
- **Bug ID**: DB-2024-001
- **Title**: Database connection pool exhaustion causing application crashes
- **Status**: `resolved`
- **Priority**: `critical`
- **Severity**: `blocker`
- **Assignee**: Backend Team
- **Reported By**: Monitoring System
- **Reported Date**: 2024-01-10
- **Updated Date**: 2024-01-11 14:30
- **Resolved Date**: 2024-01-11
- **Related Issues**: PERF-2024-005

## Description

### What Happened
Application servers began crashing intermittently under moderate load (~100 concurrent users). Errors showed "Timeout waiting for connection from pool" followed by complete service unavailability.

### Expected Behavior
Database connections should be properly managed with connection pooling, allowing the application to handle up to 500 concurrent users as specified in requirements.

### Actual Behavior
After 30 minutes of normal operation, connection pool exhaustion occurred, leading to cascading failures across all application servers.

Error messages:
```
2024-01-10 15:32:45 ERROR [main] o.a.tomcat.jdbc.pool.ConnectionPool - Unable to create initial connections of pool.
Timeout waiting for connection from pool
org.postgresql.util.PSQLException: FATAL: sorry, too many clients already
```

## Reproduction

### Steps to Reproduce
1. Start application with default configuration
2. Simulate 100 concurrent users making API calls
3. Run for 30+ minutes
4. Observe connection pool errors in logs
5. Application becomes unresponsive

### Reproduction Rate
- [x] Always (100%)

### Preconditions
- PostgreSQL database with max_connections = 100
- Application configured with connection pool size = 50
- Moderate to high user load

## Environment

### System Information
- **OS**: Ubuntu 20.04 LTS
- **Application Server**: Tomcat 9.0
- **Load Balancer**: Nginx

### Software Versions
- **Application Version**: 2.1.0
- **Framework**: Spring Boot 2.7.0
- **Database Driver**: PostgreSQL JDBC 42.5.0
- **Connection Pool**: Tomcat JDBC Pool 9.0.0

### Configuration
- **maxActive**: 50
- **maxIdle**: 10
- **minIdle**: 5
- **testOnBorrow**: true

## Investigation

### Initial Analysis
Suspected connection leak where connections are not being returned to the pool.

### Debugging Process
| Date | Action | Findings |
|------|--------|----------|
| 2024-01-10 | Enabled connection pool logging | Found connections not being closed |
| 2024-01-10 | Added connection tracking | Identified specific service methods |
| 2024-01-11 | Code review | Found missing try-with-resources |

### Logs & Error Messages
```
2024-01-10 15:32:45 WARN  [http-nio-8080-exec-12] c.z.h.HikariPool - HikariPool-1 - Connection is not available, request timed out after 30000ms.
2024-01-10 15:32:46 ERROR [http-nio-8080-exec-12] c.e.c.ExampleController - Failed to process request
org.springframework.dao.DataAccessResourceFailureException: Failed to obtain JDBC Connection
```

## Root Cause Analysis

### Root Cause
The `UserDataService.processBatch()` method was not closing database connections in exception scenarios. When exceptions occurred during batch processing, connections remained open and were never returned to the pool.

### Contributing Factors
- Lack of try-with-resources for Connection objects
- No connection lifecycle management in service layer
- Insufficient connection validation in pool configuration
- Missing connection timeout settings

### Impact Assessment
- **Users Affected**: All active users during outage (estimated 150 users)
- **Business Impact**: 2-hour service outage, potential revenue loss
- **Technical Debt**: Poor connection management pattern throughout codebase

## Fix

### Solution Implemented
1. Updated `UserDataService.processBatch()` to use try-with-resources
2. Added connection validation to pool configuration
3. Implemented connection leak detection

### Code Changes
```java
// BEFORE:
public void processBatch(List<User> users) {
    Connection conn = dataSource.getConnection();
    // ... processing logic
    // Connection not always closed on exceptions
}

// AFTER:
public void processBatch(List<User> users) {
    try (Connection conn = dataSource.getConnection();
         PreparedStatement stmt = conn.prepareStatement(SQL)) {
        // ... processing logic
    } // Connection automatically closed
}
```

### Files Modified
- `UserDataService.java`: Added try-with-resources for all database operations
- `application.properties`: Updated connection pool configuration
- `DatabaseConfig.java`: Added connection validation settings

### Pull Request/Commit
- **PR #**: 245
- **Commit Hash**: a1b2c3d4e5f6
- **Branch**: fix/connection-leak

## Verification

### Testing Performed
- [x] Unit tests added/updated
- [x] Integration tests
- [x] Manual testing
- [x] Regression testing
- [x] Load testing

### Test Results
Load testing with 200 concurrent users for 2 hours showed stable connection pool usage (max 35 connections).

### Verification Steps
1. Deploy fix to staging environment
2. Run load test with 200 concurrent users
3. Monitor connection pool metrics for 2 hours
4. Verify no connection leaks
5. Check application responsiveness

## Prevention

### How to Prevent Similar Issues
1. Implement code review checklist for resource management
2. Add connection leak detection to CI/CD pipeline
3. Use static analysis tools to detect unclosed resources

### Recommended Improvements
1. Implement connection pool monitoring dashboard
2. Add automated alerts for connection pool usage >80%
3. Create resource management guidelines for team

### Documentation Updates
- Updated "Database Best Practices" guide
- Added connection management examples to onboarding docs
```

## Example 2: Medium UI Rendering Issue

```markdown
# Bug Report: UI-2024-015

## Basic Information
- **Bug ID**: UI-2024-015
- **Title**: Dashboard charts not rendering on mobile Safari
- **Status**: `resolved`
- **Priority**: `medium`
- **Severity**: `major`
- **Assignee**: Frontend Team
- **Reported By**: Customer Support
- **Reported Date**: 2024-01-05
- **Updated Date**: 2024-01-07 11:20
- **Resolved Date**: 2024-01-07
- **Related Issues**: None

## Description

### What Happened
Dashboard charts fail to render on iOS Safari browsers (iPhone/iPad). Users see blank space where charts should appear.

### Expected Behavior
Charts should render correctly on all supported browsers including mobile Safari.

### Actual Behavior
Charts container displays blank white space. Console shows JavaScript errors related to canvas rendering.

Error in console:
```
TypeError: null is not an object (evaluating 'canvas.getContext')
```

## Reproduction

### Steps to Reproduce
1. Open dashboard on iPhone Safari
2. Navigate to Analytics section
3. Observe blank space where charts should be
4. Check browser console for errors

### Reproduction Rate
- [x] Always (100% on affected devices)

### Preconditions
- iOS 15+ Safari browser
- Dashboard with chart.js v3.8.0
- Internet connection

## Environment

### System Information
- **OS**: iOS 15.0+
- **Browser**: Safari Mobile
- **Device**: iPhone 12+, iPad Air/Pro

### Software Versions
- **Chart Library**: Chart.js 3.8.0
- **Framework**: React 18.2.0
- **Build Tool**: Webpack 5.75.0

## Investigation

### Debugging Process
| Date | Action | Findings |
|------|--------|----------|
| 2024-01-05 | Tested on iOS simulator | Confirmed issue |
| 2024-01-06 | Checked Chart.js compatibility | Found known issue with iOS |
| 2024-01-07 | Tested alternative solutions | Found workaround |

## Root Cause Analysis

### Root Cause
Chart.js uses `window.requestAnimationFrame` which has timing issues on iOS Safari when combined with React's strict mode and component lifecycle.

### Contributing Factors
- iOS Safari's aggressive power saving features
- Chart.js animation timing
- React strict mode double-rendering

## Fix

### Solution Implemented
Added `redraw()` method call after component mount and visibility check for iOS devices.

### Code Changes
```javascript
// Added to chart component:
useEffect(() => {
  if (isIOS()) {
    // Force redraw for iOS Safari
    setTimeout(() => {
      if (chartRef.current) {
        chartRef.current.update();
      }
    }, 100);
  }
}, []);
```

## Verification

### Testing Performed
- [x] Manual testing on multiple iOS devices
- [x] Cross-browser testing
- [x] Responsive design testing

### Verification Steps
1. Test on iPhone 13 Safari
2. Test on iPad Pro Safari
3. Verify charts render correctly
4. Check no regressions on other browsers
```

## Example 3: Low Priority Cosmetic Issue

```markdown
# Bug Report: CSS-2024-003

## Basic Information
- **Bug ID**: CSS-2024-003
- **Title**: Typo in error message dialog
- **Status**: `resolved`
- **Priority**: `low`
- **Severity**: `trivial`
- **Assignee**: Junior Developer
- **Reported By**: QA Team
- **Reported Date**: 2024-01-03
- **Updated Date**: 2024-01-03 16:45
- **Resolved Date**: 2024-01-03
- **Related Issues**: None

## Description

### What Happened
Error message dialog shows "An unexcepted error occurred" with typo in "unexpected".

### Expected Behavior
Error message should read "An unexpected error occurred".

### Actual Behavior
Typo in error message text.

## Fix

### Solution Implemented
Fixed spelling in error message constant.

### Code Changes
```javascript
// BEFORE:
const ERROR_MSG = "An unexcepted error occurred";

// AFTER:
const ERROR_MSG = "An unexpected error occurred";
```

## Verification

### Testing Performed
- [x] Visual verification
- [x] String search for other typos

### Verification Steps
1. Trigger error condition
2. Verify correct spelling in dialog
```

## Template Variations

### Quick Bug Report (Minimal)
```markdown
# Bug: [Brief Title]

**Status**: `open`
**Priority**: `medium`
**Reported**: YYYY-MM-DD

**Description**: [What happened]

**Steps to Reproduce**:
1. [Step 1]
2. [Step 2]

**Fix**: [Solution or workaround]
```

### Security Bug Report
```markdown
# Security Bug: [Vulnerability Title]

**Severity**: `critical`
**Confidential**: Yes
**Disclosure Date**: [Planned disclosure date]

**Vulnerability**: [Type - XSS, SQLi, etc.]
**Impact**: [What attackers could do]
**CVSS Score**: [If applicable]

**Proof of Concept**: [Steps to exploit]

**Mitigation**: [Temporary fix]
**Permanent Fix**: [Long-term solution]
```

### Performance Bug Report
```markdown
# Performance Issue: [Title]

**Metric Affected**: [Load time, memory, CPU]
**Baseline**: [Expected performance]
**Current**: [Actual performance]
**Degradation**: [Percentage difference]

**Profiling Data**: [Metrics from profiling tools]

**Bottleneck**: [Identified slow component]
**Optimization**: [Proposed improvement]
```