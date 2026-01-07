# Troubleshooting Guide

## Common Issues and Solutions

### Environment Issues

#### 1. Environment Activation Fails
**Symptoms**: `source .venv/bin/activate` doesn't work or has no effect

**Checklist**:
```bash
# 1. Verify .venv directory exists
ls -la .venv/

# 2. Check Python executable exists
ls -la .venv/bin/python  # Linux/macOS
dir .venv\Scripts\python.exe  # Windows

# 3. Verify activation script exists
ls -la .venv/bin/activate  # Linux/macOS
dir .venv\Scripts\activate.bat  # Windows

# 4. Check file permissions
chmod +x .venv/bin/activate  # Linux/macOS if needed
```

**Solutions**:
```bash
# Recreate environment
rm -rf .venv
uv venv

# On Windows, try different activation
.venv\Scripts\activate.bat
.venv\Scripts\Activate.ps1  # PowerShell

# Manual activation (Linux/macOS)
export VIRTUAL_ENV="$(pwd)/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
```

#### 2. Wrong Python Version in Environment
**Symptoms**: Environment uses different Python version than expected

**Diagnosis**:
```bash
# Check Python version in environment
source .venv/bin/activate
python --version

# Check which Python is being used
which python  # Linux/macOS
where python  # Windows
```

**Solutions**:
```bash
# Create environment with specific Python version
uv venv --python 3.11

# List available Python versions
uv python list

# Download specific version if missing
uv python download 3.11.5
```

#### 3. Environment Corruption
**Symptoms**: Packages fail to import, weird errors

**Solutions**:
```bash
# 1. Backup and recreate
cp -r .venv .venv-backup
rm -rf .venv
uv venv
uv sync

# 2. Repair existing environment
uv sync --reinstall

# 3. Check for path issues
echo $PATH  # Ensure .venv/bin is first
```

### Installation Issues

#### 1. Package Installation Fails
**Symptoms**: `uv add package` fails with various errors

**Common Error Messages and Fixes**:

**A. Network Errors**
```
ERROR: Could not find a version that satisfies the requirement
ERROR: No matching distribution found
```
```bash
# Try different index URL
uv config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"

# Check network connectivity
curl -I https://pypi.org

# Use timeout and retry
uv add package --timeout 60 --retry 3

# Offline mode (if you have cached packages)
uv add --offline package
```

**B. Build Errors**
```
error: command 'gcc' failed with exit status 1
Microsoft Visual C++ 14.0 or greater is required
```
```bash
# Install build dependencies
# Linux:
sudo apt-get install python3-dev build-essential

# macOS:
brew install openssl readline sqlite3 xz zlib

# Windows: Install Visual Studio Build Tools
# Or use pre-built wheels
uv add package --no-binary :none:

# Try different version
uv add "package<problematic-version"
```

**C. Permission Errors**
```
PermissionError: [Errno 13] Permission denied
OSError: [Errno 30] Read-only file system
```
```bash
# Check permissions
ls -la .venv/

# Fix permissions
chmod -R u+w .venv/  # Linux/macOS

# Run as admin if needed (Windows)
# Open terminal as Administrator

# Use user install location
uv config set global.target /path/to/user/cache
```

**D. SSL/TLS Errors**
```
CERTIFICATE_VERIFY_FAILED
SSL: CERTIFICATE_VERIFY_FAILED
```
```bash
# Update certificates
# macOS:
/Applications/Python\ 3.*/Install\ Certificates.command

# Disable verification (not recommended for production)
uv config set global.trusted-host pypi.org

# Use HTTP instead of HTTPS
uv config set global.index-url "http://pypi.org/simple"
```

#### 2. Dependency Resolution Failures
**Symptoms**: `uv sync` fails with dependency conflicts

**Debugging**:
```bash
# Show dependency tree
uv pip list --tree

# Check for conflicts
uv pip check

# Try different resolution strategies
uv sync --resolution=highest
uv sync --resolution=lowest-direct
uv sync --resolution=lowest

# Generate resolution report
uv pip compile pyproject.toml --dry-run --verbose
```

**Solutions**:
```bash
# 1. Update all packages
uv update

# 2. Remove conflicting package
uv remove conflicting-package

# 3. Use version constraints
uv add "package>=1.0.0,<2.0.0"

# 4. Create constraints file
echo "conflicting-package<2.0.0" > constraints.txt
uv sync -c constraints.txt

# 5. Use dependency groups
# Install only compatible groups
uv sync --group core --no-dev
```

#### 3. Slow Installation
**Symptoms**: uv is unusually slow

**Optimization**:
```bash
# 1. Enable parallel downloads
uv config set download.concurrency 20

# 2. Use connection pooling
uv config set download.pool-connections true
uv config set download.pool-maxsize 10

# 3. Configure cache
uv config set cache.dir /fast/disk/path
uv config set cache.max-size "10GB"

# 4. Use local mirror
uv config set global.index-url "http://local-mirror/pypi"

# 5. Pre-download packages
uv pip download -r requirements.txt -d ./packages
uv sync --find-links ./packages
```

### Platform-Specific Issues

#### Windows Issues
**A. Path Length Limitations**
```
OSError: [Errno 36] File name too long
```
```bash
# Enable long paths in Windows
# Group Policy: Computer Configuration > Administrative Templates > System > Filesystem
# Or Registry: HKLM\SYSTEM\CurrentControlSet\Control\FileSystem LongPathsEnabled = 1

# Use shorter path
uv venv C:\short\.venv
```

**B. Anti-virus Interference**
```
Timeout or permission errors during installation
```
- Add Python and uv to antivirus exclusion list
- Temporarily disable antivirus during installation
- Use Windows Defender controlled folder access exceptions

**C. PowerShell Execution Policy**
```
File cannot be loaded because running scripts is disabled
```
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or run activation script directly
.\.venv\Scripts\Activate.ps1 -ExecutionPolicy Bypass
```

#### macOS Issues
**A. System Integrity Protection (SIP)**
```
Permission denied for /usr/bin/python
```
- Don't install to system Python, use virtual environments
- Use `uv venv` in user directory

**B. Architecture Issues (Apple Silicon)**
```
Wheels for wrong architecture
```
```bash
# Check architecture
uname -m  # arm64 for Apple Silicon

# Use universal2 or arm64 wheels
uv add "package; sys_platform == 'darwin' and platform_machine == 'arm64'"

# Install Rosetta version if needed
arch -x86_64 uv venv
```

**C. Missing System Libraries**
```
error: can't find headers for openssl, libffi, etc.
```
```bash
# Install with Homebrew
brew install openssl readline sqlite3 xz zlib

# Set compilation flags
export LDFLAGS="-L$(brew --prefix openssl)/lib"
export CPPFLAGS="-I$(brew --prefix openssl)/include"
```

#### Linux Issues
**A. Missing Development Headers**
```
fatal error: Python.h: No such file or directory
```
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# CentOS/RHEL/Fedora
sudo yum install python3-devel gcc

# Alpine
apk add python3-dev gcc musl-dev
```

**B. GLIBC Version Issues**
```
version `GLIBC_2.29' not found
```
- Use older Linux distribution or Docker
- Compile from source
- Use manylinux wheels

**C. SELinux Restrictions**
```
Permission denied even with correct permissions
```
```bash
# Check SELinux status
getenforce

# Temporarily disable (if safe)
sudo setenforce 0

# Or add proper context
chcon -R -t httpd_sys_content_t .venv/
```

### Performance Issues

#### 1. High Memory Usage
**Symptoms**: uv uses excessive memory

**Solutions**:
```bash
# Limit memory usage
uv config set install.memory-limit "2GB"

# Use lighter resolver
uv config set install.resolver "backtracking"

# Install packages sequentially
uv sync --sequential
```

#### 2. Slow Dependency Resolution
**Symptoms**: uv hangs during resolution

**Debugging**:
```bash
# Enable verbose logging
uv sync -vv

# Check which package is problematic
uv pip compile requirements.in --dry-run --verbose

# Try alternative resolver
uv sync --resolver backjumping
```

**Solutions**:
```bash
# 1. Reduce package count
# Split requirements into multiple files

# 2. Use pre-compiled requirements
uv pip compile requirements.in -o requirements.txt --generate-hashes

# 3. Skip optional dependencies
uv add package --no-deps
```

#### 3. Disk Space Issues
**Symptoms**: Not enough disk space for cache

**Management**:
```bash
# Check cache usage
uv cache info

# Clean cache
uv cache clean

# Prune unused files
uv cache prune

# Change cache location
uv config set cache.dir /large/disk/path
```

### Integration Issues

#### 1. IDE Integration Problems
**Symptoms**: IDE doesn't recognize environment

**VS Code**:
```json
// settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.venvPath": "${workspaceFolder}/.venv"
}
```

**PyCharm/IntelliJ**:
1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select Existing environment
4. Navigate to `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows)

**Jupyter**:
```bash
# Install ipykernel in environment
uv add ipykernel

# Register kernel
uv run python -m ipykernel install --user --name=.venv

# List kernels
jupyter kernelspec list
```

#### 2. Docker Issues
**Symptoms**: Problems in Docker containers

**Common Issues**:
```dockerfile
# 1. Cache not preserved between builds
# Add cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && uv sync --frozen

# 2. Platform mismatch
# Specify platform
FROM --platform=linux/amd64 python:3.11-slim

# 3. Missing system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
```

#### 3. CI/CD Issues
**Symptoms**: Tests fail in CI but work locally

**Debugging**:
```bash
# Reproduce locally with same environment
docker run -it python:3.11-slim bash

# Check for platform differences
python -c "import platform; print(platform.platform())"

# Verify locked dependencies
uv sync --frozen --strict
```

**GitHub Actions Tips**:
```yaml
# Cache uv cache
- name: Cache uv cache
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('**/uv.lock') }}
    restore-keys: |
      ${{ runner.os }}-uv-
```

### Advanced Troubleshooting

#### 1. Debugging uv Itself
```bash
# Enable debug logging
uv --verbose add package
uv --debug sync

# Check uv configuration
uv config list
uv doctor

# Update uv
uv self update

# Report issue
uv bug-report
```

#### 2. Analyzing Logs
```bash
# Save logs to file
uv sync 2>&1 | tee install.log

# Search for errors
grep -i error install.log
grep -B5 -A5 "failed" install.log

# Analyze timing
grep "Downloading\|Installing" install.log
```

#### 3. Creating Minimal Reproduction
```bash
# Create test environment
mkdir test-issue && cd test-issue
uv init

# Create minimal pyproject.toml
cat > pyproject.toml << EOF
[project]
name = "test-issue"
version = "0.1.0"
dependencies = [
    "problematic-package",
]
EOF

# Try to reproduce
uv venv
uv sync -vv 2>&1 | tee repro.log
```

#### 4. Comparing Environments
```bash
# Export environment specs
uv pip freeze > env1.txt
# On another machine/environment
uv pip freeze > env2.txt

# Compare
diff env1.txt env2.txt

# Or use dedicated tool
pip install pip-compare
pip-compare env1.txt env2.txt
```

### Preventive Measures

#### 1. Regular Maintenance
```bash
# Weekly maintenance script
#!/bin/bash
# Update uv
uv self update

# Update packages
uv update

# Clean cache
uv cache prune

# Check for security vulnerabilities
uv pip audit

# Run tests
uv run pytest
```

#### 2. Environment Validation
```bash
# Create validation script
cat > validate_env.sh << 'EOF'
#!/bin/bash
set -e

echo "Validating environment..."
python --version
uv --version

# Check critical packages
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"

# Run basic tests
uv run pytest tests/test_basic.py -v

echo "Environment validation passed"
EOF
chmod +x validate_env.sh
```

#### 3. Backup Strategy
```bash
# Backup environment
backup_env() {
    local backup_dir="env-backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    cp pyproject.toml uv.lock requirements*.txt "$backup_dir/"
    uv pip freeze > "$backup_dir/full-freeze.txt"
    echo "Backup created: $backup_dir"
}

# Restore from backup
restore_env() {
    local backup_dir="$1"
    cp "$backup_dir/pyproject.toml" .
    cp "$backup_dir/uv.lock" .
    uv sync --frozen
}
```