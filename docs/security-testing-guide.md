# Security Scanning System Testing Guide

This document provides a comprehensive testing plan for the security vulnerability scanning system implemented in tf-shell.

## Overview

The security scanning system includes multiple tools and processes:
- **Safety**: Python dependency vulnerability scanning
- **Bandit**: Python code security analysis
- **Semgrep**: Static analysis security scanning
- **pip-audit**: Package vulnerability auditing
- **CodeQL**: GitHub's semantic code analysis
- **Dependabot**: Automated dependency updates
- **TruffleHog**: Secret scanning

## Testing Prerequisites

### 1. Install Security Tools

```bash
# Install security scanning dependencies
pip install -r requirements-security.txt

# Verify installations
safety --version
bandit --version
semgrep --version
pip-audit --version
```

### 2. Set Up Test Environment

```bash
# Create a test branch
git checkout -b test/security-scanning

# Ensure clean working directory
git status
```

## Manual Testing Procedures

### Test 1: Local Security Scan Execution

**Objective**: Verify that the security scanning script runs successfully.

**Steps**:
1. Run the security scanner:
   ```bash
   python tools/security_scan.py --output-dir test-security-reports
   ```

2. Verify output directory creation:
   ```bash
   ls -la test-security-reports/
   ```

3. Check for expected report files:
   - `security_summary_*.json`
   - `bandit_*.json`
   - `safety_*.json`
   - `semgrep_*.json`
   - `pip_audit_*.json`

**Expected Results**:
- Script completes without errors
- All report files are generated
- Summary report contains scan statistics

### Test 2: Bazel Integration Test

**Objective**: Verify Bazel can execute the security scanner.

**Steps**:
1. Run via Bazel:
   ```bash
   bazel run //tools:security_scanner -- --output-dir bazel-security-reports
   ```

2. Verify execution and output generation

**Expected Results**:
- Bazel successfully builds and runs the security scanner
- Reports are generated in the specified directory

### Test 3: Individual Tool Testing

**Objective**: Test each security tool independently.

#### Test 3a: Safety (Dependency Scanning)
```bash
# Test Safety on requirements files
safety check -r requirements_3_10.txt --json
```

#### Test 3b: Bandit (Code Security)
```bash
# Test Bandit on Python code
bandit -r tf_shell/ -f json
```

#### Test 3c: Semgrep (Static Analysis)
```bash
# Test Semgrep security rules
semgrep --config=auto tf_shell/ tf_shell_ml/
```

#### Test 3d: pip-audit (Package Audit)
```bash
# Test pip-audit
pip-audit --format=json
```

### Test 4: Configuration File Testing

**Objective**: Verify configuration files work correctly.

**Steps**:
1. Test Bandit configuration:
   ```bash
   bandit -r tf_shell/ -c .bandit
   ```

2. Verify exclusions and test selections work as expected

**Expected Results**:
- Test files are excluded from scanning
- Only specified security tests are executed

### Test 5: Error Handling Testing

**Objective**: Test system behavior with missing tools or invalid inputs.

**Steps**:
1. Test with missing tool:
   ```bash
   # Temporarily rename a tool
   mv $(which safety) $(which safety).bak
   python tools/security_scan.py
   mv $(which safety).bak $(which safety)
   ```

2. Test with invalid directory:
   ```bash
   python tools/security_scan.py --output-dir /invalid/path
   ```

**Expected Results**:
- Graceful error handling
- Informative error messages
- Non-zero exit codes for failures

## GitHub Actions Testing

### Test 6: CI/CD Pipeline Testing

**Objective**: Verify the security workflow runs correctly in GitHub Actions.

**Steps**:
1. Create a test PR with security scanning changes
2. Monitor the security workflow execution
3. Verify all jobs complete successfully
4. Check for security reports in workflow artifacts

**Expected Results**:
- All security scanning jobs pass
- Reports are uploaded as artifacts
- PR comments include security summary (if applicable)

### Test 7: Dependabot Testing

**Objective**: Verify Dependabot configuration works correctly.

**Steps**:
1. Wait for Dependabot to create dependency update PRs
2. Verify PRs have correct labels and formatting
3. Check that security updates are prioritized

**Expected Results**:
- Dependabot creates weekly update PRs
- Security updates are properly labeled
- PRs follow configured commit message format

## Security Test Cases

### Test 8: Vulnerability Detection Testing

**Objective**: Verify the system can detect known vulnerabilities.

**Steps**:
1. Temporarily add a package with known vulnerabilities to requirements
2. Run security scans
3. Verify vulnerabilities are detected and reported

**Example vulnerable packages for testing**:
```
# Add to requirements-security.txt temporarily for testing
django==1.11.0  # Known to have security vulnerabilities
requests==2.6.0  # Older version with known issues
```

**Expected Results**:
- Safety detects vulnerable dependencies
- Reports include vulnerability details and remediation advice

### Test 9: Code Security Issue Detection

**Objective**: Test detection of insecure code patterns.

**Steps**:
1. Create a test file with intentionally insecure code:
   ```python
   # test_insecure.py
   import os
   password = "hardcoded_password"  # Should trigger B105
   os.system("rm -rf /")  # Should trigger B605
   ```

2. Run Bandit scan
3. Verify issues are detected

**Expected Results**:
- Bandit detects hardcoded passwords
- Bandit detects dangerous system calls
- Issues are properly categorized by severity

## Performance Testing

### Test 10: Scan Performance Testing

**Objective**: Measure scanning performance and resource usage.

**Steps**:
1. Time the security scanning process:
   ```bash
   time python tools/security_scan.py
   ```

2. Monitor resource usage during scans
3. Test with different project sizes

**Expected Results**:
- Scans complete within reasonable time limits (< 10 minutes)
- Memory usage remains within acceptable bounds
- Performance is consistent across runs

## Regression Testing

### Test 11: Baseline Security Report

**Objective**: Establish baseline security posture for regression testing.

**Steps**:
1. Run complete security scan on clean main branch
2. Save reports as baseline
3. Compare future scans against baseline

**Expected Results**:
- Baseline reports show current security status
- New vulnerabilities are detected in subsequent scans
- False positives are minimized

## Test Automation

### Test 12: Automated Test Suite

Create automated tests for the security scanning system:

```python
# tests/test_security_scanner.py
import unittest
import subprocess
import json
from pathlib import Path

class TestSecurityScanner(unittest.TestCase):
    def test_security_scanner_execution(self):
        """Test that security scanner runs without errors."""
        result = subprocess.run([
            "python", "tools/security_scan.py", 
            "--output-dir", "test-reports"
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertTrue(Path("test-reports").exists())
    
    def test_report_generation(self):
        """Test that all expected reports are generated."""
        # Implementation details...
        pass
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Tool Installation Failures**:
   - Ensure Python version compatibility
   - Check for system dependencies
   - Use virtual environments

2. **Permission Errors**:
   - Verify write permissions for output directory
   - Check file system permissions

3. **Network Issues**:
   - Verify internet connectivity for vulnerability databases
   - Check proxy settings if applicable

4. **False Positives**:
   - Review and update tool configurations
   - Add appropriate exclusions
   - Document accepted risks

## Reporting and Documentation

### Test Results Documentation

For each test run, document:
- Test environment details
- Tool versions used
- Test execution results
- Any issues encountered
- Remediation actions taken

### Security Metrics Tracking

Track the following metrics over time:
- Number of vulnerabilities detected
- Time to remediation
- False positive rates
- Scan execution time
- Tool effectiveness

## Conclusion

This testing guide ensures the security scanning system is robust, reliable, and effective at detecting vulnerabilities in the tf-shell project. Regular execution of these tests will maintain the security posture and catch regressions early.
