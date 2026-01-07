# Debugging Guide

## Systematic Debugging Approach

### 1. Problem Identification
- **Reproduce the issue**: Make the bug happen consistently
- **Isolate the problem**: Determine scope and boundaries
- **Gather evidence**: Collect error messages, logs, screenshots

### 2. Information Collection
- **Error messages**: Copy complete error text
- **Stack traces**: Note function call hierarchy
- **Logs**: Check application and system logs
- **Environment**: Document OS, browser, versions
- **Timing**: When does it happen? How often?

### 3. Hypothesis Formation
- **Brainstorm possibilities**: List potential causes
- **Prioritize**: Start with most likely causes
- **Create testable hypotheses**: "If X is the cause, then Y should happen"

### 4. Testing Hypotheses
- **Add logging**: Instrument code to trace execution
- **Use debuggers**: Step through code execution
- **Modify variables**: Test with different inputs
- **Isolate components**: Test parts independently

### 5. Root Cause Analysis
- **Ask "why" repeatedly**: Drill down to fundamental cause
- **Consider multiple factors**: Rarely just one cause
- **Document findings**: Keep detailed notes

### 6. Solution Implementation
- **Design fix**: Consider multiple approaches
- **Test thoroughly**: Verify fix doesn't break other things
- **Document**: Update bug report with solution

## Common Debugging Techniques

### Logging Strategies
- **Strategic logging**: Add logs at key decision points
- **Log levels**: Use DEBUG, INFO, WARN, ERROR appropriately
- **Context information**: Include timestamps, user IDs, request IDs
- **Structured logging**: Use JSON format for machine parsing

### Debugger Usage
- **Breakpoints**: Stop execution at specific lines
- **Watch expressions**: Monitor variable values
- **Call stack**: Examine function call hierarchy
- **Step through**: Execute line by line

### Code Inspection
- **Code review**: Fresh eyes often spot issues
- **Static analysis**: Use tools to find potential problems
- **Compare with working code**: Identify differences
- **Check recent changes**: What changed before bug appeared?

### Testing Techniques
- **Unit tests**: Isolate and test individual components
- **Integration tests**: Test interactions between components
- **Regression tests**: Ensure fix doesn't break existing functionality
- **Edge case testing**: Test boundary conditions

## Common Error Patterns

### Null Pointer/Reference Errors
- **Symptoms**: Crashes with "null pointer" or "undefined" errors
- **Common causes**: Uninitialized variables, missing returns, race conditions
- **Debugging**: Check variable initialization, add null checks, trace data flow

### Memory Issues
- **Symptoms**: Slow performance, crashes, out of memory errors
- **Common causes**: Memory leaks, large object retention, inefficient algorithms
- **Debugging**: Use memory profilers, check for unclosed resources, analyze heap dumps

### Concurrency Issues
- **Symptoms**: Intermittent failures, race conditions, deadlocks
- **Common causes**: Shared state, improper synchronization, thread safety issues
- **Debugging**: Add synchronization, use thread-safe data structures, analyze thread dumps

### Configuration Errors
- **Symptoms**: Works in one environment but not another
- **Common causes**: Missing environment variables, incorrect settings, path issues
- **Debugging**: Compare environment configurations, check file permissions, verify paths

### Data Issues
- **Symptoms**: Incorrect calculations, missing data, corruption
- **Common causes**: Bad input data, encoding problems, rounding errors
- **Debugging**: Validate input data, check data transformations, add data logging

## Tools and Commands

### System Monitoring
```bash
# Process monitoring
top, htop, ps aux

# Memory usage
free -h, vmstat

# Disk usage
df -h, du -sh

# Network
netstat, ss, ping, traceroute
```

### Log Analysis
```bash
# View logs
tail -f /var/log/app.log
less /var/log/app.log

# Search logs
grep "ERROR" app.log
grep -A 5 -B 5 "exception" app.log

# Analyze logs
awk '{print $1}' app.log | sort | uniq -c | sort -nr
```

### Debugging Tools by Language

#### Python
```python
# Built-in debugging
import pdb; pdb.set_trace()

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profiling
python -m cProfile script.py
```

#### JavaScript/Node.js
```javascript
// Console debugging
console.log(), console.error(), console.trace()

// Debugger
debugger; // statement

// Node.js inspection
node --inspect script.js
```

#### Java
```java
// Logging
import java.util.logging.Logger;
Logger logger = Logger.getLogger(MyClass.class.getName());

// Debugging flags
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005

// Memory analysis
jmap, jstack, jstat
```

## Root Cause Analysis Techniques

### 5 Whys Method
1. **Why did the application crash?** Connection pool exhausted.
2. **Why was the connection pool exhausted?** Connections not being returned.
3. **Why were connections not returned?** Exception handling didn't close connections.
4. **Why didn't exception handling close connections?** No try-with-resources or finally block.
5. **Why no proper resource management?** Missing code review for resource handling.

### Fishbone Diagram (Ishikawa)
- **Categories**: People, Process, Technology, Environment, Materials
- **Analysis**: Identify contributing factors in each category
- **Example**: Technology → Code → Resource Management → Missing finally blocks

### Timeline Analysis
- **Create timeline**: Document events leading to failure
- **Identify patterns**: Look for recurring events or conditions
- **Correlation**: Match events with system changes or external factors

## Prevention Strategies

### Code Quality
- **Code reviews**: Multiple eyes catch issues
- **Static analysis**: Automated code quality checks
- **Testing**: Comprehensive test coverage
- **Documentation**: Clear comments and design documents

### Monitoring and Alerting
- **Application metrics**: Monitor performance indicators
- **Error tracking**: Centralized error collection
- **Alerting**: Proactive notifications for issues
- **Dashboards**: Real-time system status

### Process Improvements
- **Incident response**: Clear procedures for handling issues
- **Post-mortems**: Learn from failures
- **Knowledge base**: Document solutions for future reference
- **Training**: Regular debugging skill development

## Psychological Aspects

### Debugging Mindset
- **Stay curious**: Approach problems with curiosity, not frustration
- **Take breaks**: Step away when stuck to gain fresh perspective
- **Collaborate**: Discuss with colleagues for new insights
- **Document**: Writing clarifies thinking

### Common Pitfalls
- **Confirmation bias**: Looking for evidence that supports your hypothesis
- **Premature conclusions**: Jumping to solutions without full understanding
- **Tunnel vision**: Focusing on one aspect while missing the bigger picture
- **Frustration**: Letting emotions cloud judgment

### Effective Communication
- **Clear descriptions**: Explain issues in simple terms
- **Visual aids**: Use diagrams, screenshots, logs
- **Structured reporting**: Follow bug report templates
- **Collaboration tools**: Use shared debugging sessions