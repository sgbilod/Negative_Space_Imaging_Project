#!/usr/bin/env python3
"""
Security Audit Demonstration Script for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This script runs comprehensive security checks on the codebase, validates
cryptographic implementations, checks for common vulnerabilities, and
generates a security audit report.

Usage:
    python security_audit_demo.py [--output report.json] [--verbose]
"""

import os
import sys
import re
import json
import hashlib
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('security_audit.log')
    ]
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Security finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(Enum):
    """Security finding categories."""
    CRYPTOGRAPHY = "cryptography"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    CORS = "cors"
    SECRETS = "secrets"
    CODE_QUALITY = "code_quality"


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    id: str
    title: str
    description: str
    severity: Severity
    category: Category
    file_path: Optional[str]
    line_number: Optional[int]
    recommendation: str
    cwe_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id
        }


@dataclass
class AuditReport:
    """Security audit report."""
    timestamp: str
    project_name: str
    version: str
    findings: List[SecurityFinding]
    summary: Dict[str, int]
    passed_checks: List[str]
    failed_checks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "project_name": self.project_name,
            "version": self.version,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks
        }


class SecurityAuditor:
    """Security audit implementation."""
    
    def __init__(self, project_root: str, verbose: bool = False):
        self.project_root = Path(project_root)
        self.verbose = verbose
        self.findings: List[SecurityFinding] = []
        self.passed_checks: List[str] = []
        self.failed_checks: List[str] = []
        self.finding_counter = 0
        
    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        self.finding_counter += 1
        return f"SEC-{self.finding_counter:04d}"
    
    def _add_finding(self, finding: SecurityFinding):
        """Add a security finding."""
        self.findings.append(finding)
        if self.verbose:
            logger.warning(f"Finding: {finding.title} ({finding.severity.value})")
    
    def _log_check(self, name: str, passed: bool):
        """Log check result."""
        if passed:
            self.passed_checks.append(name)
            if self.verbose:
                logger.info(f"✅ PASSED: {name}")
        else:
            self.failed_checks.append(name)
            if self.verbose:
                logger.warning(f"❌ FAILED: {name}")
    
    def run_audit(self) -> AuditReport:
        """Run comprehensive security audit."""
        logger.info("Starting security audit...")
        logger.info(f"Project root: {self.project_root}")
        
        # Run all checks
        self.check_cors_configuration()
        self.check_jwt_security()
        self.check_cryptographic_implementations()
        self.check_hardcoded_secrets()
        self.check_input_validation()
        self.check_sql_injection_patterns()
        self.check_xss_patterns()
        self.check_path_traversal_patterns()
        self.check_security_headers()
        self.check_rate_limiting()
        self.check_dependency_security()
        self.check_authentication_mechanisms()
        self.check_rbac_implementation()
        self.check_session_security()
        self.check_encryption_usage()
        
        # Generate summary
        summary = {
            "total_findings": len(self.findings),
            "critical": sum(1 for f in self.findings if f.severity == Severity.CRITICAL),
            "high": sum(1 for f in self.findings if f.severity == Severity.HIGH),
            "medium": sum(1 for f in self.findings if f.severity == Severity.MEDIUM),
            "low": sum(1 for f in self.findings if f.severity == Severity.LOW),
            "info": sum(1 for f in self.findings if f.severity == Severity.INFO),
            "checks_passed": len(self.passed_checks),
            "checks_failed": len(self.failed_checks)
        }
        
        report = AuditReport(
            timestamp=datetime.utcnow().isoformat() + "Z",
            project_name="Negative Space Imaging Project",
            version="1.0.0",
            findings=self.findings,
            summary=summary,
            passed_checks=self.passed_checks,
            failed_checks=self.failed_checks
        )
        
        logger.info(f"Audit complete: {summary['total_findings']} findings")
        return report
    
    def check_cors_configuration(self):
        """Check CORS configuration for security issues."""
        logger.info("Checking CORS configuration...")
        
        # Check for wildcard origins
        patterns = [
            (r'origin:\s*["\']?\*["\']?', "Wildcard CORS origin"),
            (r'allow_origins\s*=\s*\[\s*["\']?\*["\']?\s*\]', "Wildcard in allow_origins"),
            (r'Access-Control-Allow-Origin.*\*', "Wildcard Access-Control header"),
        ]
        
        wildcard_found = False
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js")) + \
                         list(self.project_root.rglob("*.yaml")) + \
                         list(self.project_root.rglob("*.yml"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc in patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Wildcard CORS Configuration: {desc}",
                            description=f"Found wildcard (*) in CORS configuration which allows any origin to access resources.",
                            severity=Severity.HIGH,
                            category=Category.CORS,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Replace wildcard with explicit list of allowed origins.",
                            cwe_id="CWE-942"
                        ))
                        wildcard_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        self._log_check("CORS Wildcard Check", not wildcard_found)
        
        # Check for proper CORS configuration
        cors_config_found = False
        for file_path in [
            self.project_root / "src" / "config" / "security.ts",
            self.project_root / "api" / "api.py",
            self.project_root / "security_config.yaml"
        ]:
            if file_path.exists():
                content = file_path.read_text(errors='ignore')
                if "cors" in content.lower() and "origin" in content.lower():
                    cors_config_found = True
                    break
        
        self._log_check("CORS Configuration Exists", cors_config_found)
    
    def check_jwt_security(self):
        """Check JWT implementation security."""
        logger.info("Checking JWT security...")
        
        issues_found = False
        
        # Patterns to check
        patterns = [
            (r'algorithm\s*[=:]\s*["\']none["\']', "JWT None Algorithm", Severity.CRITICAL),
            (r'verify\s*[=:]\s*False', "JWT Verification Disabled", Severity.CRITICAL),
            (r'JWT_SECRET\s*=\s*["\'][^"\']+["\']', "Hardcoded JWT Secret", Severity.HIGH),
            (r'expiresIn\s*[=:]\s*["\']?0["\']?', "Zero JWT Expiry", Severity.HIGH),
        ]
        
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc, severity in patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"JWT Security Issue: {desc}",
                            description=f"Found potential JWT security issue: {desc}",
                            severity=severity,
                            category=Category.AUTHENTICATION,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Use strong algorithm (HS256/RS256), enable verification, use environment variables for secrets.",
                            cwe_id="CWE-347"
                        ))
                        issues_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        # Check for proper JWT secret handling
        jwt_env_check = False
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if re.search(r'(os\.getenv|process\.env|os\.environ)\s*[\(\[]?\s*["\']JWT_SECRET', content):
                    jwt_env_check = True
                    break
            except Exception:
                pass
        
        self._log_check("JWT Secret from Environment", jwt_env_check)
        self._log_check("JWT Implementation Security", not issues_found)
    
    def check_cryptographic_implementations(self):
        """Validate cryptographic implementations."""
        logger.info("Checking cryptographic implementations...")
        
        weak_crypto_found = False
        
        # Weak cryptography patterns - focused on security-critical contexts
        weak_patterns = [
            (r'\bMD5\b(?!.*#\s*nosec)', "MD5 Hash", "Use SHA-256 or stronger"),
            (r'\bSHA1\b|\bSHA-1\b', "SHA-1 Hash", "Use SHA-256 or stronger"),
            (r'\bDES\b[^3]', "DES Encryption", "Use AES-256"),
            (r'\bRC4\b|\bARC4\b', "RC4 Stream Cipher", "Use AES-GCM"),
        ]
        
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path) or "test" in str(file_path).lower():
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc, recommendation in weak_patterns:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        # Skip if it's in a comment or test context
                        line = content.split('\n')[line_num - 1]
                        match_text = match.group()
                        match_pos = line.find(match_text)
                        # Only check for comment prefix if match was found in line
                        if match_pos >= 0 and ('#' in line[:match_pos] or '//' in line[:match_pos]):
                            continue
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Weak Cryptography: {desc}",
                            description=f"Found potentially weak cryptographic implementation: {desc}",
                            severity=Severity.MEDIUM,
                            category=Category.CRYPTOGRAPHY,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation=recommendation,
                            cwe_id="CWE-327"
                        ))
                        weak_crypto_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        # Check for strong crypto usage
        strong_crypto_found = False
        for file_path in self.project_root.rglob("*.py"):
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if any(x in content for x in ['AES', 'ChaCha20', 'HKDF', 'SHA256', 'SHA512']):
                    strong_crypto_found = True
                    break
            except Exception:
                pass
        
        self._log_check("Strong Cryptography Used", strong_crypto_found)
        self._log_check("No Weak Cryptography", not weak_crypto_found)
    
    def check_hardcoded_secrets(self):
        """Check for hardcoded secrets and credentials."""
        logger.info("Checking for hardcoded secrets...")
        
        secrets_found = False
        
        # Secret patterns (conservative to reduce false positives)
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\'\s]{8,}["\']', "Hardcoded Password"),
            (r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{16,}["\']', "Hardcoded API Key"),
            (r'secret[_-]?key\s*=\s*["\'][^"\'\s]{16,}["\']', "Hardcoded Secret Key"),
            (r'aws[_-]?access[_-]?key[_-]?id\s*=\s*["\']AK[A-Z0-9]{18}["\']', "AWS Access Key"),
            (r'PRIVATE[_-]?KEY.*BEGIN.*PRIVATE.*KEY', "Private Key in Code"),
        ]
        
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js")) + \
                         list(self.project_root.rglob("*.json"))
        
        # Exclude example/template files
        exclude_patterns = ['.example', '.template', '.sample', 'test', 'mock']
        
        for file_path in files_to_check:
            if any(p in str(file_path).lower() for p in exclude_patterns):
                continue
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc in secret_patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE | re.DOTALL))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Potential Hardcoded Secret: {desc}",
                            description=f"Found potential hardcoded secret: {desc}",
                            severity=Severity.HIGH,
                            category=Category.SECRETS,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Use environment variables or a secrets management system.",
                            cwe_id="CWE-798"
                        ))
                        secrets_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        self._log_check("No Hardcoded Secrets", not secrets_found)
    
    def check_input_validation(self):
        """Check for input validation mechanisms."""
        logger.info("Checking input validation...")
        
        # Check for validation libraries
        validation_found = False
        validation_files = [
            self.project_root / "package.json",
            self.project_root / "requirements.txt"
        ]
        
        validation_libs = ['joi', 'zod', 'yup', 'validator', 'pydantic', 'marshmallow', 'cerberus']
        
        for file_path in validation_files:
            if file_path.exists():
                content = file_path.read_text(errors='ignore')
                if any(lib in content.lower() for lib in validation_libs):
                    validation_found = True
                    break
        
        if not validation_found:
            self._add_finding(SecurityFinding(
                id=self._generate_finding_id(),
                title="Input Validation Library Missing",
                description="No common input validation library detected in dependencies.",
                severity=Severity.MEDIUM,
                category=Category.INPUT_VALIDATION,
                file_path=None,
                line_number=None,
                recommendation="Add input validation library (e.g., Joi, Zod for JS; Pydantic for Python).",
                cwe_id="CWE-20"
            ))
        
        self._log_check("Input Validation Library", validation_found)
    
    def check_sql_injection_patterns(self):
        """Check for SQL injection vulnerabilities."""
        logger.info("Checking for SQL injection patterns...")
        
        sql_issues_found = False
        
        # SQL injection patterns
        patterns = [
            (r'execute\s*\(\s*["\'][^"\']*%[sd][^"\']*["\']', "String Formatting in SQL"),
            (r'execute\s*\(\s*f["\']', "F-String in SQL"),
            (r'query\s*\(\s*["\'][^"\']*\+', "String Concatenation in Query"),
            (r'\.raw\s*\(\s*["\'][^"\']*\$\{', "Template Literal in Raw Query"),
        ]
        
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path) or "test" in str(file_path).lower():
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc in patterns:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Potential SQL Injection: {desc}",
                            description=f"Found potential SQL injection vulnerability: {desc}",
                            severity=Severity.HIGH,
                            category=Category.INPUT_VALIDATION,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Use parameterized queries or ORM methods.",
                            cwe_id="CWE-89"
                        ))
                        sql_issues_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        self._log_check("No SQL Injection Patterns", not sql_issues_found)
    
    def check_xss_patterns(self):
        """Check for XSS vulnerabilities."""
        logger.info("Checking for XSS patterns...")
        
        xss_issues_found = False
        
        # XSS patterns
        patterns = [
            (r'innerHTML\s*=', "innerHTML Assignment"),
            (r'document\.write\s*\(', "document.write Usage"),
            (r'dangerouslySetInnerHTML', "React dangerouslySetInnerHTML"),
            (r'v-html\s*=', "Vue v-html Directive"),
        ]
        
        files_to_check = list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js")) + \
                         list(self.project_root.rglob("*.tsx")) + \
                         list(self.project_root.rglob("*.jsx")) + \
                         list(self.project_root.rglob("*.vue"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc in patterns:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Potential XSS: {desc}",
                            description=f"Found potential XSS vulnerability: {desc}",
                            severity=Severity.MEDIUM,
                            category=Category.INPUT_VALIDATION,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Sanitize user input before rendering. Use textContent instead of innerHTML.",
                            cwe_id="CWE-79"
                        ))
                        xss_issues_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        # Check for XSS protection
        xss_protection_found = False
        for file_path in self.project_root.rglob("package.json"):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if "xss-clean" in content or "xss" in content:
                    xss_protection_found = True
                    break
            except Exception:
                pass
        
        self._log_check("XSS Protection Library", xss_protection_found)
        self._log_check("No Direct XSS Patterns", not xss_issues_found)
    
    def check_path_traversal_patterns(self):
        """Check for path traversal vulnerabilities."""
        logger.info("Checking for path traversal patterns...")
        
        path_issues_found = False
        
        # Path traversal patterns
        patterns = [
            (r'open\s*\([^)]*\+[^)]*\)', "Dynamic File Path"),
            (r'readFile\s*\([^)]*\+[^)]*\)', "Dynamic File Read"),
            (r'writeFile\s*\([^)]*\+[^)]*\)', "Dynamic File Write"),
            (r'path\.join\s*\([^)]*req\.[^)]*\)', "User Input in Path"),
        ]
        
        files_to_check = list(self.project_root.rglob("*.py")) + \
                         list(self.project_root.rglob("*.ts")) + \
                         list(self.project_root.rglob("*.js"))
        
        for file_path in files_to_check:
            if "node_modules" in str(file_path) or ".git" in str(file_path) or "test" in str(file_path).lower():
                continue
            try:
                content = file_path.read_text(errors='ignore')
                for pattern, desc in patterns:
                    matches = list(re.finditer(pattern, content))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self._add_finding(SecurityFinding(
                            id=self._generate_finding_id(),
                            title=f"Potential Path Traversal: {desc}",
                            description=f"Found potential path traversal vulnerability: {desc}",
                            severity=Severity.HIGH,
                            category=Category.INPUT_VALIDATION,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            recommendation="Validate and sanitize file paths. Use allowlist for permitted paths.",
                            cwe_id="CWE-22"
                        ))
                        path_issues_found = True
            except Exception as e:
                if self.verbose:
                    logger.debug(f"Error reading {file_path}: {e}")
        
        self._log_check("No Path Traversal Patterns", not path_issues_found)
    
    def check_security_headers(self):
        """Check for security header configurations."""
        logger.info("Checking security headers configuration...")
        
        # Check for Helmet.js usage
        helmet_found = False
        package_json = self.project_root / "package.json"
        if package_json.exists():
            content = package_json.read_text(errors='ignore')
            if "helmet" in content:
                helmet_found = True
        
        self._log_check("Helmet.js Security Headers", helmet_found)
        
        # Check for security header configuration
        headers_config_found = False
        for file_path in list(self.project_root.rglob("*.ts")) + list(self.project_root.rglob("*.js")):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if "contentSecurityPolicy" in content or "hsts" in content:
                    headers_config_found = True
                    break
            except Exception:
                pass
        
        self._log_check("Security Headers Configured", headers_config_found)
    
    def check_rate_limiting(self):
        """Check for rate limiting implementation."""
        logger.info("Checking rate limiting...")
        
        rate_limit_found = False
        
        # Check package.json for rate limiting
        package_json = self.project_root / "package.json"
        if package_json.exists():
            content = package_json.read_text(errors='ignore')
            if "express-rate-limit" in content or "rate-limit" in content:
                rate_limit_found = True
        
        # Check Python requirements
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            content = requirements.read_text(errors='ignore')
            if "slowapi" in content or "flask-limiter" in content:
                rate_limit_found = True
        
        self._log_check("Rate Limiting Implemented", rate_limit_found)
    
    def check_dependency_security(self):
        """Check dependency security."""
        logger.info("Checking dependency security...")
        
        # Check if npm audit would find issues (don't run, just check if package-lock exists)
        package_lock = self.project_root / "package-lock.json"
        npm_audit_available = package_lock.exists()
        self._log_check("NPM Lock File Exists", npm_audit_available)
        
        # Check if Python requirements.txt exists
        requirements = self.project_root / "requirements.txt"
        self._log_check("Python Requirements Defined", requirements.exists())
    
    def check_authentication_mechanisms(self):
        """Check authentication mechanisms."""
        logger.info("Checking authentication mechanisms...")
        
        # Check for bcrypt or argon2
        secure_hash_found = False
        package_json = self.project_root / "package.json"
        if package_json.exists():
            content = package_json.read_text(errors='ignore')
            if "bcrypt" in content or "argon2" in content:
                secure_hash_found = True
        
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            content = requirements.read_text(errors='ignore')
            if "bcrypt" in content or "argon2" in content or "passlib" in content:
                secure_hash_found = True
        
        self._log_check("Secure Password Hashing", secure_hash_found)
        
        # Check for 2FA implementation
        twofa_found = False
        for file_path in list(self.project_root.rglob("*.py")) + list(self.project_root.rglob("*.ts")):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if "two_factor" in content.lower() or "2fa" in content.lower() or "totp" in content.lower():
                    twofa_found = True
                    break
            except Exception:
                pass
        
        self._log_check("Two-Factor Authentication", twofa_found)
    
    def check_rbac_implementation(self):
        """Check RBAC implementation."""
        logger.info("Checking RBAC implementation...")
        
        rbac_found = False
        for file_path in list(self.project_root.rglob("*.py")) + list(self.project_root.rglob("*.ts")):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                content_lower = content.lower()
                # Check for RBAC or role-based permission patterns
                if "rbac" in content_lower or ("role" in content_lower and "permission" in content_lower):
                    rbac_found = True
                    break
            except Exception:
                pass
        
        self._log_check("RBAC Implementation", rbac_found)
    
    def check_session_security(self):
        """Check session security configuration."""
        logger.info("Checking session security...")
        
        session_security_found = False
        for file_path in list(self.project_root.rglob("*.ts")) + list(self.project_root.rglob("*.js")):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                # Check for secure session configuration
                if "httpOnly" in content and "secure" in content and "sameSite" in content:
                    session_security_found = True
                    break
            except Exception:
                pass
        
        self._log_check("Secure Session Configuration", session_security_found)
    
    def check_encryption_usage(self):
        """Check encryption usage in the project."""
        logger.info("Checking encryption usage...")
        
        encryption_found = False
        for file_path in list(self.project_root.rglob("*.py")):
            if "node_modules" in str(file_path):
                continue
            try:
                content = file_path.read_text(errors='ignore')
                if "cryptography" in content or "encryption" in content.lower():
                    encryption_found = True
                    break
            except Exception:
                pass
        
        self._log_check("Encryption Implementation", encryption_found)


def print_report(report: AuditReport):
    """Print audit report to console."""
    print("\n" + "=" * 70)
    print("SECURITY AUDIT REPORT")
    print("=" * 70)
    print(f"\nProject: {report.project_name}")
    print(f"Version: {report.version}")
    print(f"Timestamp: {report.timestamp}")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total Findings: {report.summary['total_findings']}")
    print(f"  - Critical: {report.summary['critical']}")
    print(f"  - High: {report.summary['high']}")
    print(f"  - Medium: {report.summary['medium']}")
    print(f"  - Low: {report.summary['low']}")
    print(f"  - Info: {report.summary['info']}")
    print(f"\nChecks Passed: {report.summary['checks_passed']}")
    print(f"Checks Failed: {report.summary['checks_failed']}")
    
    if report.passed_checks:
        print("\n" + "-" * 70)
        print("PASSED CHECKS")
        print("-" * 70)
        for check in report.passed_checks:
            print(f"  ✅ {check}")
    
    if report.failed_checks:
        print("\n" + "-" * 70)
        print("FAILED CHECKS")
        print("-" * 70)
        for check in report.failed_checks:
            print(f"  ❌ {check}")
    
    if report.findings:
        print("\n" + "-" * 70)
        print("FINDINGS")
        print("-" * 70)
        for finding in sorted(report.findings, key=lambda f: ["critical", "high", "medium", "low", "info"].index(f.severity.value)):
            severity_icons = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
                "info": "⚪"
            }
            icon = severity_icons.get(finding.severity.value, "⚪")
            print(f"\n{icon} [{finding.id}] {finding.title}")
            print(f"   Severity: {finding.severity.value.upper()}")
            print(f"   Category: {finding.category.value}")
            if finding.file_path:
                print(f"   File: {finding.file_path}:{finding.line_number}")
            print(f"   Description: {finding.description}")
            print(f"   Recommendation: {finding.recommendation}")
            if finding.cwe_id:
                print(f"   CWE: {finding.cwe_id}")
    
    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Security Audit Demo for Negative Space Imaging Project"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).parent),
        help="Project root directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Do not exit with error code on findings"
    )
    
    args = parser.parse_args()
    
    # Run audit
    auditor = SecurityAuditor(args.project_root, args.verbose)
    report = auditor.run_audit()
    
    # Print report to console
    print_report(report)
    
    # Save JSON report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report saved to: {args.output}")
    
    # Exit with error code if critical or high findings (unless --no-fail is set)
    if not args.no_fail and (report.summary['critical'] > 0 or report.summary['high'] > 0):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
