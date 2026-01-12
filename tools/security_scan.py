#!/usr/bin/env python3
"""
Security vulnerability scanning script for tf-shell project.

This script runs various security scans and generates a comprehensive report.
It can be used by developers locally or in CI/CD pipelines.

Usage:
    python tools/security_scan.py [--output-dir OUTPUT_DIR] [--format FORMAT]

Requirements:
    pip install -r requirements-security.txt
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SecurityScanner:
    """Main security scanner class that orchestrates different security tools."""
    
    def __init__(self, output_dir: str = "security-reports", format_type: str = "json"):
        """
        Initialize the security scanner.
        
        Args:
            output_dir: Directory to store security reports
            format_type: Output format (json, txt, html)
        """
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str, str]:
        """
        Run a shell command and return the result.
        
        Args:
            command: Command to run as list of strings
            description: Description of the command for logging
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        print(f"Running {description}...")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out: {' '.join(command)}"
        except Exception as e:
            return False, "", f"Error running command: {e}"
    
    def scan_python_dependencies(self) -> Dict:
        """Scan Python dependencies for known vulnerabilities using Safety."""
        print("\n=== Python Dependency Vulnerability Scan ===")
        results = {"tool": "safety", "findings": [], "errors": []}
        
        # Find all requirements files
        req_files = list(Path(".").glob("requirements*.txt"))
        
        for req_file in req_files:
            print(f"Scanning {req_file}...")
            
            # Run safety check
            command = ["safety", "check", "-r", str(req_file), "--json"]
            success, stdout, stderr = self.run_command(command, f"Safety scan for {req_file}")
            
            output_file = self.output_dir / f"safety_{req_file.stem}_{self.timestamp}.json"
            
            if success and stdout:
                try:
                    safety_data = json.loads(stdout)
                    results["findings"].extend(safety_data)
                    
                    # Save detailed report
                    with open(output_file, 'w') as f:
                        json.dump(safety_data, f, indent=2)
                        
                except json.JSONDecodeError:
                    results["errors"].append(f"Failed to parse Safety output for {req_file}")
            else:
                results["errors"].append(f"Safety scan failed for {req_file}: {stderr}")
        
        return results
    
    def scan_python_code(self) -> Dict:
        """Scan Python code for security issues using Bandit."""
        print("\n=== Python Code Security Scan ===")
        results = {"tool": "bandit", "findings": [], "errors": []}
        
        # Directories to scan
        scan_dirs = ["tf_shell", "tf_shell_ml", "tools"]
        existing_dirs = [d for d in scan_dirs if Path(d).exists()]
        
        if not existing_dirs:
            results["errors"].append("No Python directories found to scan")
            return results
        
        # Run Bandit scan
        command = [
            "bandit", "-r", *existing_dirs,
            "-f", "json",
            "-o", str(self.output_dir / f"bandit_{self.timestamp}.json")
        ]
        
        success, stdout, stderr = self.run_command(command, "Bandit security scan")
        
        if success:
            try:
                # Read the output file
                with open(self.output_dir / f"bandit_{self.timestamp}.json", 'r') as f:
                    bandit_data = json.load(f)
                    results["findings"] = bandit_data.get("results", [])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                results["errors"].append(f"Failed to read Bandit output: {e}")
        else:
            results["errors"].append(f"Bandit scan failed: {stderr}")
        
        return results
    
    def scan_with_semgrep(self) -> Dict:
        """Run Semgrep static analysis security scanner."""
        print("\n=== Semgrep Static Analysis Scan ===")
        results = {"tool": "semgrep", "findings": [], "errors": []}
        
        # Run Semgrep with security rules
        command = [
            "semgrep", "--config=auto", "--json",
            "--output", str(self.output_dir / f"semgrep_{self.timestamp}.json"),
            "tf_shell/", "tf_shell_ml/"
        ]
        
        success, stdout, stderr = self.run_command(command, "Semgrep security scan")
        
        if success:
            try:
                with open(self.output_dir / f"semgrep_{self.timestamp}.json", 'r') as f:
                    semgrep_data = json.load(f)
                    results["findings"] = semgrep_data.get("results", [])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                results["errors"].append(f"Failed to read Semgrep output: {e}")
        else:
            results["errors"].append(f"Semgrep scan failed: {stderr}")
        
        return results
    
    def audit_pip_packages(self) -> Dict:
        """Audit installed pip packages for vulnerabilities."""
        print("\n=== Pip Package Audit ===")
        results = {"tool": "pip-audit", "findings": [], "errors": []}
        
        command = ["pip-audit", "--format=json", "--output", 
                  str(self.output_dir / f"pip_audit_{self.timestamp}.json")]
        
        success, stdout, stderr = self.run_command(command, "Pip package audit")
        
        if success:
            try:
                with open(self.output_dir / f"pip_audit_{self.timestamp}.json", 'r') as f:
                    audit_data = json.load(f)
                    results["findings"] = audit_data.get("vulnerabilities", [])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                results["errors"].append(f"Failed to read pip-audit output: {e}")
        else:
            results["errors"].append(f"Pip audit failed: {stderr}")
        
        return results
    
    def generate_summary_report(self, scan_results: List[Dict]) -> None:
        """Generate a summary report of all security scans."""
        print("\n=== Generating Security Summary Report ===")
        
        summary = {
            "scan_timestamp": datetime.now().isoformat(),
            "project": "tf-shell",
            "total_scans": len(scan_results),
            "scans": scan_results,
            "summary": {
                "total_findings": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0,
                "errors": 0
            }
        }
        
        # Count findings and errors
        for scan in scan_results:
            summary["summary"]["total_findings"] += len(scan.get("findings", []))
            summary["summary"]["errors"] += len(scan.get("errors", []))
            
            # Count severity levels (implementation depends on tool output format)
            for finding in scan.get("findings", []):
                severity = finding.get("severity", "").lower()
                if "high" in severity:
                    summary["summary"]["high_severity"] += 1
                elif "medium" in severity:
                    summary["summary"]["medium_severity"] += 1
                else:
                    summary["summary"]["low_severity"] += 1
        
        # Save summary report
        summary_file = self.output_dir / f"security_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print(f"\n=== Security Scan Summary ===")
        print(f"Total scans completed: {summary['total_scans']}")
        print(f"Total findings: {summary['summary']['total_findings']}")
        print(f"High severity: {summary['summary']['high_severity']}")
        print(f"Medium severity: {summary['summary']['medium_severity']}")
        print(f"Low severity: {summary['summary']['low_severity']}")
        print(f"Errors: {summary['summary']['errors']}")
        print(f"\nDetailed reports saved to: {self.output_dir}")
        
        # Return non-zero exit code if high severity issues found
        if summary['summary']['high_severity'] > 0:
            print("\n⚠️  HIGH SEVERITY VULNERABILITIES FOUND!")
            print("Please review and address these issues before proceeding.")
            return 1
        elif summary['summary']['medium_severity'] > 0:
            print("\n⚠️  Medium severity vulnerabilities found.")
            print("Consider addressing these issues.")
        
        return 0
    
    def run_all_scans(self) -> int:
        """Run all security scans and generate reports."""
        print("Starting comprehensive security scan for tf-shell...")
        print(f"Reports will be saved to: {self.output_dir}")
        
        scan_results = []
        
        # Run all scans
        try:
            scan_results.append(self.scan_python_dependencies())
            scan_results.append(self.scan_python_code())
            scan_results.append(self.scan_with_semgrep())
            scan_results.append(self.audit_pip_packages())
        except KeyboardInterrupt:
            print("\nScan interrupted by user.")
            return 1
        except Exception as e:
            print(f"Unexpected error during scanning: {e}")
            return 1
        
        # Generate summary report
        return self.generate_summary_report(scan_results)


def main():
    """Main entry point for the security scanner."""
    parser = argparse.ArgumentParser(
        description="Run security vulnerability scans for tf-shell project"
    )
    parser.add_argument(
        "--output-dir",
        default="security-reports",
        help="Directory to store security reports (default: security-reports)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "txt", "html"],
        default="json",
        help="Output format for reports (default: json)"
    )
    
    args = parser.parse_args()
    
    # Check if required tools are installed
    required_tools = ["safety", "bandit", "semgrep", "pip-audit"]
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Error: Missing required security tools: {', '.join(missing_tools)}")
        print("Please install them with: pip install -r requirements-security.txt")
        return 1
    
    # Run security scans
    scanner = SecurityScanner(args.output_dir, args.format)
    return scanner.run_all_scans()


if __name__ == "__main__":
    sys.exit(main())
