# Security Layer

This directory contains security-related frameworks and implementations for the 
Negative Space Imaging Project.

Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

## Overview

The security layer provides comprehensive protection for the imaging system,
implementing defense-in-depth strategies, HIPAA compliance requirements, and
quantum-resistant cryptography.

## Components

### 1. Encryption Systems

#### quantum_encryption.py
- Implements quantum-resistant encryption using ChaCha20-Poly1305
- ECDH key exchange with NIST P-521 curve
- HKDF-SHA512 for key derivation
- Automatic key rotation every 24 hours

#### encryption-service.yaml
- Kubernetes deployment configuration for encryption service
- HashiCorp Vault integration for key management

### 2. Access Control

#### rbac.py
- Role-Based Access Control (RBAC) implementation
- Thread-safe user and role management
- Bootstrap mechanism for first admin user
- Role assignment and permission checking

Supported Roles:
- `admin`: Full system access
- `analyst`: Image analysis and reporting
- `user`: Basic access to own resources
- `viewer`: Read-only access

### 3. Authentication Mechanisms

#### biometric_auth.py
- Multi-factor authentication support
- Biometric verification integration
- Session management
- Failed attempt tracking and lockout

### 4. Security Monitoring

#### security_monitor.py
- Real-time security event monitoring
- Event correlation and pattern detection
- Alert generation for suspicious activity
- Audit log management with retention policies

#### audit_logging.py
- Comprehensive audit trail
- HIPAA-compliant logging
- Tamper-evident log storage
- Log integrity verification

## Configuration Files

### security-policy.yaml
Kubernetes security policy configuration including:
- Quantum encryption settings
- Network policies
- Pod security configurations
- Audit policy rules

### quantum-resistant.yaml
Quantum-resistant algorithm configuration:
- CRYSTALS-Kyber for key encapsulation
- CRYSTALS-Dilithium for digital signatures

### vault-config.yaml
HashiCorp Vault configuration for secrets management

## Security Configuration

The main security configuration is defined in `/security_config.yaml` which includes:

### CORS Configuration
- Explicit allowed origins (no wildcards in production)
- Strict method and header control
- Credentials support with proper CSRF protection

### Rate Limiting
- Global: 100 requests per 15 minutes
- Auth endpoints: 10 requests per 5 minutes
- API endpoints: 60 requests per minute
- Upload endpoints: 20 requests per hour

### Authentication
- JWT tokens with HS256 algorithm
- 1-hour access token expiry
- 7-day refresh token expiry
- Secure password policy (12+ characters, complexity requirements)

### Encryption
- AES-256-GCM for symmetric encryption
- ECDH with secp521r1 for key exchange
- ChaCha20-Poly1305 for quantum-resistant encryption
- TLS 1.2+ required for data in transit

## Security Best Practices

### Environment Variables
All sensitive configuration must be set via environment variables:

```bash
# Required environment variables
JWT_SECRET=<64-character-hex-string>
ENCRYPTION_KEY=<64-character-hex-string>
ENCRYPTION_IV=<16-character-hex-string>
COOKIE_SECRET=<random-string>
```

Generate secure values:
```bash
# Generate JWT secret
openssl rand -hex 32

# Generate encryption key
openssl rand -hex 32

# Generate IV
openssl rand -hex 8
```

### CORS Security
Never use wildcard (`*`) origins in production. Always specify explicit domains:

```typescript
cors: {
  origin: ['https://negative-space-imaging.com'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE']
}
```

### Session Security
```typescript
session: {
  httpOnly: true,      // Prevent XSS access
  secure: true,        // HTTPS only
  sameSite: 'strict',  // CSRF protection
  maxAge: 3600000      // 1 hour
}
```

## Security Testing

### Running Security Audit
```bash
python security_audit_demo.py --verbose --output report.json
```

### Running Security Tests
```bash
# Python security tests
pytest tests/test_security.py -v

# Sovereign security integration tests
python sovereign/security_test.py
```

### Dependency Scanning
```bash
# NPM audit
npm audit

# Python safety check
pip install safety
safety check -r requirements.txt
```

## Incident Response

### Alert Thresholds
- Failed logins: 10 per minute triggers alert
- API errors: 50 per minute triggers alert
- Suspicious requests: 100 per hour triggers alert

### Automatic Responses
- Brute force: IP blocked for 30 minutes after 5 attempts
- Rate limit abuse: IP blocked for 60 minutes after 100 violations

## HIPAA Compliance

The security layer implements HIPAA technical safeguards:

### Access Controls
- Unique user identification
- Automatic logoff after inactivity
- Encryption of PHI at rest and in transit

### Audit Controls
- Hardware and software activity logging
- 6-year audit log retention

### Integrity Controls
- Authentication mechanisms
- Data integrity verification via cryptographic hashes

### Transmission Security
- TLS 1.2+ encryption required
- End-to-end encryption for sensitive data

## Files in This Directory

| File | Description |
|------|-------------|
| `audit_logging.py` | Comprehensive audit logging system |
| `biometric_auth.py` | Biometric authentication implementation |
| `encryption-service.yaml` | Kubernetes encryption service config |
| `quantum-resistant.yaml` | Quantum-resistant algorithm config |
| `quantum_encryption.py` | Quantum-enhanced encryption module |
| `rbac.py` | Role-Based Access Control implementation |
| `security-policy.yaml` | Security policy definitions |
| `security_monitor.py` | Real-time security monitoring |
| `vault-config.yaml` | HashiCorp Vault configuration |
| `SECURITY_HARDENING.md` | Security hardening guidelines |

## Related Files

- `/security_config.yaml` - Main security configuration
- `/security_audit_demo.py` - Security audit demonstration script
- `/tests/test_security.py` - Security unit tests
- `/sovereign/security_test.py` - Security integration tests
- `/src/config/security.ts` - TypeScript security configuration
- `/src/middleware/security.ts` - Express security middleware

## Contact

For security concerns or vulnerability reports, contact:
security@negative-space-imaging.com
