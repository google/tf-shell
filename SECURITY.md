# Security Policy

## Supported Versions

We actively support the following versions of tf-shell with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in tf-shell, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities to:
- **Email**: security@google.com
- **Subject**: [tf-shell] Security Vulnerability Report

### What to Include

When reporting a vulnerability, please include:

1. **Description**: A clear description of the vulnerability
2. **Impact**: Potential impact and attack scenarios
3. **Reproduction**: Step-by-step instructions to reproduce the issue
4. **Environment**: 
   - tf-shell version
   - Python version
   - Operating system
   - TensorFlow version
5. **Proof of Concept**: If applicable, include a minimal proof of concept
6. **Suggested Fix**: If you have ideas for fixing the vulnerability

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Within 7 days with preliminary assessment
- **Resolution**: Security patches will be released as soon as possible after verification

### Security Best Practices

When using tf-shell in production:

1. **Keep Dependencies Updated**: Regularly update tf-shell and its dependencies
2. **Secure Key Management**: 
   - Never hardcode encryption keys in source code
   - Use secure key storage mechanisms
   - Rotate keys regularly
3. **Input Validation**: Always validate inputs before encryption/decryption
4. **Secure Communication**: Use TLS for network communication
5. **Access Control**: Implement proper access controls for encrypted data
6. **Audit Logging**: Log security-relevant events for monitoring

### Vulnerability Disclosure Policy

1. We will acknowledge receipt of vulnerability reports within 48 hours
2. We will provide regular updates on the progress of fixing the vulnerability
3. We will publicly disclose vulnerabilities after they have been fixed
4. We may coordinate disclosure with other affected parties if necessary
5. We will credit reporters in our security advisories (unless they prefer to remain anonymous)

### Security Contacts

For security-related questions or concerns:
- **Security Team**: security@google.com
- **Project Maintainers**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)

### Hall of Fame

We recognize and thank security researchers who responsibly disclose vulnerabilities:

<!-- Security researchers who have contributed will be listed here -->

---

This security policy is based on industry best practices and may be updated as needed.
