# Common Errors and Solutions

## Python Errors

### ImportError
**Error Message**: `ImportError: No module named 'module_name'`
**Common Causes**:
- Module not installed
- Virtual environment not activated
- Incorrect PYTHONPATH
- Typo in import statement

**Solutions**:
```bash
# Install missing module
pip install module_name

# Check if in virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Check Python path
python -c "import sys; print(sys.path)"
```

### ModuleNotFoundError
**Error Message**: `ModuleNotFoundError: No module named 'module'`
**Similar to ImportError** but more specific to Python 3.6+

**Solutions**:
```bash
# Install package
pip install package-name

# For local modules, ensure __init__.py exists
touch module/__init__.py

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/module"
```

### SyntaxError
**Error Message**: `SyntaxError: invalid syntax`
**Common Causes**:
- Missing parentheses, brackets, or quotes
- Incorrect indentation
- Using Python 2 syntax in Python 3
- Invalid characters

**Examples & Fixes**:
```python
# Missing closing parenthesis
# WRONG: print("Hello"
# FIX: print("Hello")

# Incorrect indentation
# WRONG:
def func():
print("Hello")
# FIX:
def func():
    print("Hello")

# Python 2 print in Python 3
# WRONG: print "Hello"
# FIX: print("Hello")
```

### IndentationError
**Error Message**: `IndentationError: unexpected indent`
**Common Causes**:
- Mixing tabs and spaces
- Inconsistent indentation levels
- Missing indentation after colon

**Solutions**:
- Configure editor to use spaces (recommended: 4 spaces)
- Run `python -m tabnanny file.py` to check for issues
- Use IDE with auto-formatting (black, autopep8)

### TypeError
**Error Message**: `TypeError: unsupported operand type(s) for +: 'int' and 'str'`
**Common Causes**:
- Incorrect data type operations
- Function called with wrong argument types
- Missing or extra arguments

**Examples & Fixes**:
```python
# String + integer
# WRONG: "Age: " + 25
# FIX: "Age: " + str(25) or f"Age: {25}"

# Function argument mismatch
def greet(name):
    return f"Hello {name}"
# WRONG: greet("John", "Doe")  # Too many arguments
# FIX: greet("John")

# Wrong argument type
# WRONG: len(123)
# FIX: len(str(123))
```

### ValueError
**Error Message**: `ValueError: invalid literal for int() with base 10: 'abc'`
**Common Causes**:
- Invalid value for conversion
- Out of range values
- Empty sequences

**Examples & Fixes**:
```python
# Invalid conversion
# WRONG: int("abc")
# FIX: 
try:
    value = int("123")
except ValueError:
    value = 0

# Empty sequence
# WRONG: max([])
# FIX:
if my_list:
    result = max(my_list)
else:
    result = None
```

### KeyError
**Error Message**: `KeyError: 'key_name'`
**Common Causes**:
- Accessing non-existent dictionary key
- Missing key in configuration

**Solutions**:
```python
# Use get() with default
value = my_dict.get('key', default_value)

# Check before access
if 'key' in my_dict:
    value = my_dict['key']

# Use defaultdict
from collections import defaultdict
my_dict = defaultdict(lambda: 'default')
```

### AttributeError
**Error Message**: `AttributeError: 'object' has no attribute 'method_name'`
**Common Causes**:
- Calling non-existent method
- Incorrect object type
- Missing imports

**Solutions**:
```python
# Check object type
print(type(obj))

# Use hasattr()
if hasattr(obj, 'method_name'):
    obj.method_name()

# Verify imports
from module import ClassName
obj = ClassName()
```

### IndexError
**Error Message**: `IndexError: list index out of range`
**Common Causes**:
- Accessing list beyond its length
- Empty list access

**Solutions**:
```python
# Check length before access
if len(my_list) > index:
    value = my_list[index]

# Use try-except
try:
    value = my_list[index]
except IndexError:
    value = None

# Use slicing (safe)
value = my_list[index:index+1]
if value:
    value = value[0]
```

## JavaScript/Node.js Errors

### TypeError: Cannot read property 'x' of undefined/null
**Common Causes**:
- Accessing nested property on undefined object
- Asynchronous code timing issues

**Solutions**:
```javascript
// Optional chaining (ES2020)
const value = obj?.property?.nested;

// Default value
const value = (obj && obj.property) || defaultValue;

// Nullish coalescing
const value = obj?.property ?? defaultValue;
```

### ReferenceError: variable is not defined
**Common Causes**:
- Using undeclared variable
- Typo in variable name
- Scope issues

**Solutions**:
```javascript
// Declare before use
let variable;
variable = value;

// Check scope
function example() {
    // variable declared here is local
    let variable = "local";
}

// Use 'use strict' to catch errors
'use strict';
undeclaredVariable = 5; // Throws ReferenceError
```

### SyntaxError: Unexpected token
**Common Causes**:
- Missing brackets, parentheses, or quotes
- Invalid JSON
- Reserved word usage

**Solutions**:
```javascript
// Check JSON validity
try {
    JSON.parse(invalidJson);
} catch (e) {
    console.log("Invalid JSON:", e.message);
}

// Validate syntax with linter
// Use prettier or eslint
```

## Database Errors

### Connection Errors
**MySQL**: `ERROR 2003 (HY000): Can't connect to MySQL server`
**PostgreSQL**: `connection refused`
**Common Causes**:
- Database server not running
- Incorrect host/port
- Firewall blocking
- Authentication failure

**Solutions**:
```bash
# Check if service is running
sudo systemctl status mysql
sudo systemctl start mysql

# Verify connection
mysql -u username -p -h hostname -P port

# Check firewall
sudo ufw status
sudo ufw allow 3306/tcp  # MySQL
sudo ufw allow 5432/tcp  # PostgreSQL
```

### Authentication Errors
**MySQL**: `ERROR 1045 (28000): Access denied for user`
**PostgreSQL**: `FATAL: password authentication failed for user`
**Solutions**:
```sql
-- Reset password
ALTER USER 'username'@'localhost' IDENTIFIED BY 'new_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON database.* TO 'username'@'localhost';
FLUSH PRIVILEGES;
```

### Deadlocks
**MySQL**: `Deadlock found when trying to get lock`
**PostgreSQL**: `deadlock detected`
**Solutions**:
- Keep transactions short
- Access tables in consistent order
- Use appropriate isolation levels
- Implement retry logic

## Web Development Errors

### CORS Errors
**Error**: `Access-Control-Allow-Origin` header missing
**Solutions**:
```javascript
// Server-side (Node.js/Express)
app.use(cors({
    origin: 'https://yourdomain.com',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true
}));

// Nginx configuration
add_header 'Access-Control-Allow-Origin' 'https://yourdomain.com';
add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
```

### 404 Not Found
**Common Causes**:
- Incorrect URL path
- Missing route handler
- File not in correct location

**Solutions**:
```python
# Flask example
@app.route('/correct-path')
def handler():
    return "Found!"

# Check file exists
import os
if os.path.exists('file.txt'):
    # Serve file
```

### 500 Internal Server Error
**Common Causes**:
- Unhandled exceptions
- Configuration errors
- Resource exhaustion

**Debugging**:
```python
# Add error logging
import logging
logging.basicConfig(level=logging.ERROR)

# Use try-except
try:
    risky_operation()
except Exception as e:
    logging.error(f"Error: {e}")
    return "Error occurred", 500
```

## Network Errors

### Connection Refused
**Common Causes**:
- Service not running on port
- Firewall blocking
- Incorrect port number

**Solutions**:
```bash
# Check if port is listening
netstat -tulpn | grep :3000
sudo lsof -i :3000

# Test connection
telnet hostname port
nc -zv hostname port
```

### Timeout Errors
**Common Causes**:
- Network latency
- Server overload
- DNS issues

**Solutions**:
```python
# Increase timeout
import requests
response = requests.get(url, timeout=30)

# Implement retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_request():
    return requests.get(url)
```

### SSL/TLS Errors
**Common Causes**:
- Expired certificates
- Self-signed certificates
- Incorrect certificate chain

**Solutions**:
```python
# Verify certificates
import ssl
context = ssl.create_default_context()

# For development (NOT production)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
response = requests.get(url, verify=False)
```

## Performance Errors

### Memory Leaks
**Symptoms**: Increasing memory usage over time
**Common Causes**:
- Unclosed resources (files, connections)
- Event listener accumulation
- Circular references

**Solutions**:
```python
# Use context managers
with open('file.txt', 'r') as f:
    content = f.read()

# Remove event listeners
element.removeEventListener('click', handler)

# Use weak references
import weakref
ref = weakref.ref(object)
```

### CPU Spikes
**Symptoms**: High CPU usage, slow response
**Common Causes**:
- Infinite loops
- Inefficient algorithms
- Excessive recursion

**Solutions**:
```python
# Profile code
import cProfile
cProfile.run('my_function()')

# Optimize algorithms
# O(n²) → O(n log n) or O(n)

# Add limits
MAX_ITERATIONS = 1000
for i in range(MAX_ITERATIONS):
    if condition:
        break
```

## Environment-Specific Errors

### Windows Path Issues
**Error**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solutions**:
```python
# Use os.path for cross-platform compatibility
import os
path = os.path.join('folder', 'subfolder', 'file.txt')

# Handle drive letters
path = r'C:\Users\Name\file.txt'  # raw string
path = 'C:/Users/Name/file.txt'   # forward slashes
```

### Linux Permission Errors
**Error**: `PermissionError: [Errno 13] Permission denied`
**Solutions**:
```bash
# Check permissions
ls -la file.txt

# Change permissions
chmod 644 file.txt  # owner: rw, group: r, others: r
chown user:group file.txt

# Run with sudo (if appropriate)
sudo python script.py
```

### macOS Specific Issues
**Error**: SSL certificate verification failures
**Solutions**:
```bash
# Install certificates
/Applications/Python\ 3.x/Install\ Certificates.command

# Update Python
brew update
brew upgrade python
```

## Prevention Strategies

### Proactive Error Handling
```python
# Use try-except appropriately
try:
    risky_operation()
except SpecificError as e:
    handle_error(e)
except AnotherError as e:
    handle_differently(e)
else:
    # Code that runs if no exception
    process_result()
finally:
    # Cleanup code (always runs)
    cleanup_resources()
```

### Input Validation
```python
def process_input(value):
    if not isinstance(value, (int, float)):
        raise TypeError("Expected number")
    if value < 0:
        raise ValueError("Must be non-negative")
    return value * 2
```

### Logging and Monitoring
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Testing
```python
import unittest

class TestErrorHandling(unittest.TestCase):
    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            process_input(-1)
    
    def test_valid_input(self):
        result = process_input(5)
        self.assertEqual(result, 10)
```