"""
SSL/TLS configuration for secure deployment.
"""

import os
import ssl
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)

class SSLConfig:
    """SSL/TLS configuration management."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.is_production = self.environment == "production"
        
    @property
    def ssl_enabled(self) -> bool:
        """Check if SSL is enabled."""
        return bool(self.ssl_keyfile and self.ssl_certfile)
    
    @property
    def ssl_keyfile(self) -> Optional[str]:
        """Get SSL private key file path."""
        keyfile = os.getenv("SSL_KEYFILE")
        if keyfile and not Path(keyfile).exists():
            logger.error(f"SSL keyfile not found: {keyfile}")
            return None
        return keyfile
    
    @property
    def ssl_certfile(self) -> Optional[str]:
        """Get SSL certificate file path."""
        certfile = os.getenv("SSL_CERTFILE")
        if certfile and not Path(certfile).exists():
            logger.error(f"SSL certfile not found: {certfile}")
            return None
        return certfile
    
    @property
    def ssl_ca_certs(self) -> Optional[str]:
        """Get SSL CA certificates file path."""
        ca_certs = os.getenv("SSL_CA_CERTS")
        if ca_certs and not Path(ca_certs).exists():
            logger.warning(f"SSL CA certs file not found: {ca_certs}")
            return None
        return ca_certs
    
    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for the server."""
        if not self.ssl_enabled:
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            
            # Load certificate and key
            context.load_cert_chain(self.ssl_certfile, self.ssl_keyfile)
            
            # Configure security settings
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            
            # Load CA certificates if provided
            if self.ssl_ca_certs:
                context.load_verify_locations(self.ssl_ca_certs)
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context.verify_mode = ssl.CERT_NONE
            
            # Security options
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE
            
            logger.info("SSL context created successfully")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            return None
    
    def get_uvicorn_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for uvicorn."""
        if not self.ssl_enabled:
            logger.warning("SSL not configured. Running without HTTPS.")
            return {}
        
        config = {
            "ssl_keyfile": self.ssl_keyfile,
            "ssl_certfile": self.ssl_certfile,
        }
        
        if self.ssl_ca_certs:
            config["ssl_ca_certs"] = self.ssl_ca_certs
        
        # SSL configuration
        config.update({
            "ssl_version": ssl.PROTOCOL_TLS,
            "ssl_cert_reqs": ssl.CERT_NONE,  # Adjust based on requirements
            "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
        })
        
        return config
    
    def validate_certificates(self) -> Dict[str, Any]:
        """Validate SSL certificates."""
        validation_result = {
            "ssl_enabled": self.ssl_enabled,
            "issues": [],
            "warnings": []
        }
        
        if not self.ssl_enabled:
            if self.is_production:
                validation_result["issues"].append("SSL certificates not configured for production")
            else:
                validation_result["warnings"].append("SSL not configured for development")
            return validation_result
        
        # Check certificate files
        if not Path(self.ssl_certfile).exists():
            validation_result["issues"].append(f"SSL certificate file not found: {self.ssl_certfile}")
        
        if not Path(self.ssl_keyfile).exists():
            validation_result["issues"].append(f"SSL private key file not found: {self.ssl_keyfile}")
        
        # Check file permissions (Unix-like systems)
        try:
            key_path = Path(self.ssl_keyfile)
            if key_path.exists():
                # Check if private key is readable by others
                key_stat = key_path.stat()
                if key_stat.st_mode & 0o077:
                    validation_result["warnings"].append("SSL private key file has overly permissive permissions")
        except Exception as e:
            logger.debug(f"Could not check SSL key permissions: {e}")
        
        # Validate certificate content (basic check)
        try:
            with open(self.ssl_certfile, 'r') as f:
                cert_content = f.read()
                if "BEGIN CERTIFICATE" not in cert_content:
                    validation_result["issues"].append("SSL certificate file appears to be invalid")
        except Exception as e:
            validation_result["issues"].append(f"Could not read SSL certificate: {e}")
        
        return validation_result

def generate_self_signed_cert(cert_dir: str = "./certs") -> Dict[str, str]:
    """
    Generate self-signed certificate for development.
    
    WARNING: Only use for development! Never use self-signed certificates in production.
    """
    cert_path = Path(cert_dir)
    cert_path.mkdir(exist_ok=True)
    
    keyfile = cert_path / "server.key"
    certfile = cert_path / "server.crt"
    
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Development"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"API Testing Framework"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.DNSName(u"127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write private key
        with open(keyfile, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write certificate
        with open(certfile, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Set secure permissions on private key
        os.chmod(keyfile, 0o600)
        
        logger.info(f"Self-signed certificate generated: {certfile}")
        logger.warning("Using self-signed certificate - not suitable for production!")
        
        return {
            "keyfile": str(keyfile),
            "certfile": str(certfile),
            "warning": "Self-signed certificate generated - not suitable for production!"
        }
        
    except ImportError:
        logger.error("cryptography library not available. Cannot generate self-signed certificate.")
        return {}
    except Exception as e:
        logger.error(f"Failed to generate self-signed certificate: {e}")
        return {}

# Global SSL configuration
ssl_config = SSLConfig()
