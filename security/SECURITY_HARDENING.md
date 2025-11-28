# Security Hardening Guide

Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

This document outlines the security measures implemented in the Negative Space Imaging Project and provides guidance for secure deployment.

## Table of Contents

- [Overview](#overview)
- [Critical Security Requirements](#critical-security-requirements)
- [Environment Variables](#environment-variables)
- [CORS Configuration](#cors-configuration)
- [JWT Authentication](#jwt-authentication)
- [Rate Limiting](#rate-limiting)
- [Security Headers](#security-headers)
- [Input Validation](#input-validation)
- [Encryption](#encryption)
- [HIPAA Compliance](#hipaa-compliance)
- [Production Deployment Checklist](#production-deployment-checklist)

## Overview

The Negative Space Imaging Project implements multiple layers of security to protect sensitive imaging data and ensure compliance with healthcare security standards (HIPAA).

## Critical Security Requirements

### 1. JWT Secret Configuration

**CRITICAL**: The JWT_SECRET environment variable MUST be set before starting the application. The application will fail to start without it.

```bash
# Generate a secure JWT secret (64+ characters recommended)
openssl rand -hex 32
```

Set in your environment:
```bash
export JWT_SECRET="your-generated-secret-here"
```

### 2. CORS Origins

**CRITICAL**: Never use wildcard (`*`) CORS origins in production. Always specify exact allowed origins.

```bash
# Set allowed origins (comma-separated for multiple)
export ALLOWED_ORIGINS="https://yourdomain.com,https://api.yourdomain.com"
```

### 3. Encryption Keys

For production deployments, encryption keys must be set:

```bash
# Generate encryption key (32 bytes = 64 hex characters)
openssl rand -hex 32

# Generate initialization vector (8 bytes = 16 hex characters)
openssl rand -hex 8
```

## Environment Variables

All security-sensitive values are configured via environment variables. See `.env.example` for a complete list.

### Required for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `JWT_SECRET` | JWT signing secret (required) | 64+ char random string |
| `ALLOWED_ORIGINS` | CORS allowed origins | `https://yourdomain.com` |
| `DATABASE_URL` | Database connection string | `postgresql://user:pass@host:5432/db` |
| `ENCRYPTION_KEY` | AES-256 encryption key | 64 hex characters |
| `ENCRYPTION_IV` | Encryption initialization vector | 16 hex characters |

### Recommended Settings for Production

| Variable | Recommended Value |
|----------|-------------------|
| `NODE_ENV` | `production` |
| `ENABLE_HTTPS_ONLY` | `true` |
| `ENABLE_HSTS` | `true` |
| `DB_SSL` | `true` |
| `BCRYPT_ROUNDS` | `12` or higher |

## CORS Configuration

### Implementation

CORS is configured in both the Python FastAPI (`api/api.py`) and Node.js Express (`src/middleware/security.ts`) components.

**FastAPI (Python)**:
```python
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Signature", "X-Timestamp"],
)
```

**Express (Node.js)**:
```typescript
cors: {
    origin: process.env.CORS_ORIGIN || ['http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    credentials: true,
}
```

### Best Practices

1. Always specify exact origins (never use `*` in production)
2. Only allow necessary HTTP methods
3. Limit allowed headers to those actually used
4. Set appropriate `max-age` for preflight caching

## JWT Authentication

### Configuration

JWT authentication is enforced across all protected API endpoints.

```python
# Python API
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable must be set")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
```

### Security Measures

- No default secret value (application fails without configuration)
- Short token expiration (30 minutes default)
- HS256 algorithm by default (configurable)
- Refresh token support for session management

## Rate Limiting

### General Rate Limiting

Applied to all API endpoints:
- Window: 15 minutes
- Max requests: 100 per IP

### Authentication Rate Limiting

Stricter limits for auth endpoints:
- Window: 5 minutes  
- Max requests: 10 per IP
- Failed login tracking with account lockout

### Implementation

```typescript
// General rate limiter
app.use(rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
}));

// Auth-specific rate limiter
const authLimiter = rateLimit({
    windowMs: 5 * 60 * 1000,
    max: 10,
    message: 'Too many authentication attempts',
});
```

## Security Headers

Helmet.js is configured with comprehensive security headers:

| Header | Setting | Purpose |
|--------|---------|---------|
| Content-Security-Policy | Restrictive | Prevent XSS and injection |
| X-Frame-Options | DENY | Prevent clickjacking |
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| Strict-Transport-Security | 1 year, includeSubDomains | Force HTTPS |
| X-XSS-Protection | Enabled | XSS filtering |
| Referrer-Policy | strict-origin-when-cross-origin | Control referrer info |

## Input Validation

### Request Validation

All API inputs are validated using:
- **Python**: Pydantic models
- **Node.js**: Joi schemas

### Size Limits

- JSON body: 1MB max
- File uploads: 10MB max (configurable)
- URL-encoded: 1MB max

### Sanitization

- XSS protection via `xss-clean` middleware
- HTTP Parameter Pollution prevention via `hpp`

## Encryption

### Data at Rest

- AES-256-CBC encryption for sensitive data
- Keys stored in environment variables

### Data in Transit

- TLS 1.3 enforced for all connections
- HSTS enabled in production
- Secure cookies with HttpOnly and SameSite flags

## HIPAA Compliance

The system is designed for HIPAA compliance with:

1. **Access Controls**: Role-based access control (RBAC)
2. **Audit Logging**: Comprehensive logging of all access
3. **Encryption**: End-to-end encryption of PHI
4. **Session Management**: Automatic logout after inactivity
5. **Multi-Factor Authentication**: Support for MFA

## Production Deployment Checklist

Before deploying to production, verify:

- [ ] `JWT_SECRET` is set to a unique, secure value (64+ characters)
- [ ] `ALLOWED_ORIGINS` contains only your production domains
- [ ] `DATABASE_URL` uses SSL connection (`sslmode=require`)
- [ ] `ENCRYPTION_KEY` and `ENCRYPTION_IV` are set
- [ ] `NODE_ENV` is set to `production`
- [ ] `ENABLE_HTTPS_ONLY` is `true`
- [ ] `ENABLE_HSTS` is `true`
- [ ] `DB_SSL` is `true`
- [ ] `BCRYPT_ROUNDS` is 12 or higher
- [ ] All API keys are real, non-placeholder values
- [ ] Default database passwords are changed
- [ ] Redis password is set if Redis is exposed
- [ ] Firewall rules restrict database/Redis access
- [ ] SSL certificates are valid and not self-signed
- [ ] Log files are secured and rotated
- [ ] Regular security scans are scheduled

## Additional Resources

- [SECURITY.md](../SECURITY.md) - Security policy and vulnerability reporting
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment instructions
- [HIPAA Compliance Documentation](./README.md)

## Contact

For security concerns, contact: security@negativespacesystems.com
