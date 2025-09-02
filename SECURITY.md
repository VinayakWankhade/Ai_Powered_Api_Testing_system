# Security Hardening Guide

## Overview

This document outlines the security improvements implemented in the AI-Powered API Testing Framework to address potential vulnerabilities and ensure production readiness.

## Security Improvements Implemented

### 1. Authentication & Authorization ✅

- **API Key Authentication**: Implemented secure API key validation
- **Rate Limited Authentication**: Prevent brute force attacks
- **Secure Key Storage**: API keys are hashed before logging
- **Optional Authentication**: Public endpoints with enhanced features for authenticated users

**Configuration:**
```bash
# Required for production
API_KEYS="key1,key2,key3"  # Comma-separated list of valid API keys
MASTER_API_KEY="your-master-key"  # Master key for admin operations
```

### 2. CORS Security ✅

- **Environment-Based CORS**: Different policies for development and production
- **Restricted Origins**: No more wildcard (*) origins in production
- **Secure Headers**: Proper CORS headers configuration

**Configuration:**
```bash
# Production CORS settings
ENVIRONMENT="production"
ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# Development automatically allows localhost
ENVIRONMENT="development"  # Allows localhost:3000, localhost:8080
```

### 3. Rate Limiting ✅

- **Endpoint-Specific Limits**: Different limits for different operations
- **User-Based Tracking**: Per-user rate limiting
- **Configurable Limits**: Environment-based rate limit configuration

**Default Limits:**
- Health checks: 10/minute
- Status checks: 5/minute  
- Spec uploads: 10/hour
- General API access: 30/minute
- Spec deletion: 5/hour

### 4. Input Validation & Sanitization ✅

- **Comprehensive Validation**: All inputs validated and sanitized
- **Size Limits**: Maximum file and request sizes enforced
- **SQL Injection Prevention**: Parameterized queries and input sanitization
- **XSS Prevention**: HTML content sanitization

**Configuration:**
```bash
MAX_SPEC_SIZE="10485760"      # 10MB max API specification size
MAX_REQUEST_SIZE="1048576"    # 1MB max request size
MAX_UPLOAD_SIZE="5242880"     # 5MB max file upload size
```

### 5. Secure Logging & Audit Trails ✅

- **Structured Logging**: JSON format for production
- **Security Audit Logs**: Separate audit log files
- **Data Privacy**: Sensitive data hashed before logging
- **Comprehensive Coverage**: Authentication, API access, security events

**Configuration:**
```bash
LOG_LEVEL="INFO"
LOG_FILE="./logs/api_testing.log"
AUDIT_LOGGING="true"
AUDIT_LOG_FILE="./logs/audit.log"
LOG_FORMAT="json"  # For production
```

### 6. AI Service Security ✅

- **Rate Limiting**: AI request rate limiting per user
- **Input Sanitization**: Prompt injection prevention
- **Output Validation**: AI response validation and filtering
- **Cost Monitoring**: Token usage and cost tracking

**Configuration:**
```bash
OPENAI_API_KEY="your-openai-key"
AI_RATE_LIMIT_PER_HOUR="1000"
```

### 7. Environment & Secrets Management ✅

- **Environment Validation**: Required variables checked at startup
- **Encryption Support**: Built-in encryption for sensitive data
- **Development vs Production**: Different configurations per environment

**Configuration:**
```bash
ENVIRONMENT="production"  # or "development"
ENCRYPTION_PASSWORD="your-secure-password"
ENCRYPTION_SALT="your-secure-salt"
```

### 8. HTTPS/TLS Support ✅

- **SSL Configuration**: Full SSL/TLS support with uvicorn
- **Security Headers**: Comprehensive security headers
- **Certificate Validation**: SSL certificate validation
- **Development Certificates**: Self-signed certificate generation for development

**Configuration:**
```bash
# SSL Certificate files
SSL_KEYFILE="/path/to/private.key"
SSL_CERTFILE="/path/to/certificate.crt"
SSL_CA_CERTS="/path/to/ca-bundle.crt"  # Optional
```

## Security Headers Implemented

The following security headers are automatically added:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: [strict policy]`

## Production Deployment Checklist

### Required Environment Variables
```bash
# Database
DATABASE_URL="postgresql://user:pass@host:port/db"

# Security
API_KEYS="key1,key2,key3"
MASTER_API_KEY="master-key"
SECRET_KEY="your-app-secret-key"
ENCRYPTION_PASSWORD="strong-encryption-password"
ENCRYPTION_SALT="unique-salt-value"

# CORS
ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# SSL
SSL_KEYFILE="/path/to/private.key"
SSL_CERTFILE="/path/to/certificate.crt"

# AI Services
OPENAI_API_KEY="your-openai-key"

# Environment
ENVIRONMENT="production"
DEBUG="false"

# Logging
LOG_LEVEL="INFO"
LOG_FILE="/var/log/api-testing/app.log"
AUDIT_LOG_FILE="/var/log/api-testing/audit.log"
LOG_FORMAT="json"
```

### Security Validation

The system automatically validates security configuration at startup:

1. **Required Variables**: Checks for missing production variables
2. **Certificate Validation**: Validates SSL certificates
3. **Security Warnings**: Logs potential security issues
4. **Rate Limit Configuration**: Validates rate limiting setup

## API Authentication

### Using API Keys

All protected endpoints require authentication:

```bash
# Using curl
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://your-api.com/api/v1/upload-spec

# Using Python requests
headers = {"Authorization": "Bearer YOUR_API_KEY"}
response = requests.post("https://your-api.com/api/v1/upload-spec", 
                        headers=headers, json=data)
```

### Generating API Keys

The system provides secure API key generation:

```python
from src.config.security_config import get_config
config = get_config()
new_key = config.generate_api_key()
print(f"New API key: {new_key}")
```

## Monitoring & Alerts

### Security Events Logged

- Authentication attempts (success/failure)
- API access patterns
- Rate limit violations
- Input validation failures
- AI service usage and errors
- Data access events

### Log Analysis

Security logs are structured for easy analysis:

```bash
# Search for authentication failures
grep "authentication.*failed" /var/log/api-testing/audit.log

# Monitor rate limit violations
grep "rate_limit_exceeded" /var/log/api-testing/audit.log

# Track AI usage
grep "AI completion" /var/log/api-testing/app.log
```

## Development vs Production

### Development Mode
- Self-signed certificates allowed
- Relaxed CORS policy (localhost)
- Detailed error messages
- Optional authentication

### Production Mode
- SSL certificates required
- Strict CORS policy
- Minimal error disclosure
- Mandatory authentication
- Enhanced logging

## Security Best Practices

### 1. Regular Security Updates
- Keep dependencies updated
- Monitor security advisories
- Regular security audits

### 2. Monitoring
- Set up log monitoring
- Configure alerting for security events
- Monitor API usage patterns

### 3. Backup & Recovery
- Regular database backups
- Secure backup storage
- Tested recovery procedures

### 4. Access Control
- Use strong API keys
- Rotate keys regularly
- Monitor access patterns

## Remaining Security Considerations

While the implemented security measures significantly improve the framework's security posture, consider these additional measures for high-security environments:

1. **WAF Integration**: Web Application Firewall
2. **DDoS Protection**: Distributed denial-of-service protection
3. **Vulnerability Scanning**: Regular automated security scans
4. **Penetration Testing**: Professional security testing
5. **Compliance**: GDPR, SOC2, or other relevant compliance requirements

## Contact

For security issues or questions, please refer to the project's security policy or contact the development team.
