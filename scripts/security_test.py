#!/usr/bin/env python3
"""
Security validation script for the AI-Powered API Testing Framework.

This script tests the implemented security measures to ensure they are working correctly.
"""

import os
import sys
import json
import time
import requests
import hashlib
from typing import Dict, Any, List
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class SecurityTester:
    """Security testing utilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    def test_cors_security(self):
        """Test CORS configuration."""
        print("\n🔒 Testing CORS Security...")
        
        # Test with unauthorized origin
        headers = {
            'Origin': 'https://malicious-site.com',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        try:
            response = self.session.options(f"{self.base_url}/api/v1/specs", headers=headers)
            
            # Check if unauthorized origin is rejected
            cors_header = response.headers.get('Access-Control-Allow-Origin')
            if cors_header and cors_header != 'https://malicious-site.com':
                self.log_test("CORS Origin Restriction", True, "Unauthorized origin rejected")
            else:
                self.log_test("CORS Origin Restriction", False, "Unauthorized origin accepted")
        except Exception as e:
            self.log_test("CORS Origin Restriction", False, f"Error: {str(e)}")
    
    def test_authentication(self):
        """Test authentication requirements."""
        print("\n🔐 Testing Authentication...")
        
        # Test without API key
        try:
            response = self.session.post(f"{self.base_url}/api/v1/upload-spec", json={
                "name": "test",
                "spec_type": "openapi",
                "specification": {"openapi": "3.0.0"}
            })
            
            if response.status_code == 401:
                self.log_test("Authentication Required", True, "Unauthorized request rejected")
            else:
                self.log_test("Authentication Required", False, f"Expected 401, got {response.status_code}")
        except Exception as e:
            self.log_test("Authentication Required", False, f"Error: {str(e)}")
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        print("\n⏱️  Testing Rate Limiting...")
        
        # Test health endpoint rate limit (10/minute)
        try:
            successful_requests = 0
            rate_limited = False
            
            for i in range(15):  # Try more than the limit
                response = self.session.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
                time.sleep(0.1)  # Small delay
            
            if rate_limited:
                self.log_test("Rate Limiting", True, f"Rate limit triggered after {successful_requests} requests")
            else:
                self.log_test("Rate Limiting", False, "Rate limit not triggered")
        except Exception as e:
            self.log_test("Rate Limiting", False, f"Error: {str(e)}")
    
    def test_input_validation(self):
        """Test input validation."""
        print("\n🛡️  Testing Input Validation...")
        
        # Test large payload rejection
        try:
            large_payload = {
                "name": "test",
                "spec_type": "openapi",
                "specification": {"data": "x" * 20000000}  # ~20MB
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/upload-spec", json=large_payload)
            
            if response.status_code in [400, 413, 422]:  # Bad Request, Payload Too Large, or Validation Error
                self.log_test("Large Payload Rejection", True, "Large payload rejected")
            else:
                self.log_test("Large Payload Rejection", False, f"Expected 4xx, got {response.status_code}")
        except Exception as e:
            # Connection errors are expected for very large payloads
            if "Connection" in str(e) or "timeout" in str(e).lower():
                self.log_test("Large Payload Rejection", True, "Large payload rejected (connection error)")
            else:
                self.log_test("Large Payload Rejection", False, f"Error: {str(e)}")
        
        # Test invalid input format
        try:
            invalid_payload = {
                "name": "../../../etc/passwd",  # Path traversal attempt
                "spec_type": "invalid_type",
                "specification": "<script>alert('xss')</script>"
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/upload-spec", json=invalid_payload)
            
            if response.status_code in [400, 422]:
                self.log_test("Input Validation", True, "Invalid input rejected")
            else:
                self.log_test("Input Validation", False, f"Expected 4xx, got {response.status_code}")
        except Exception as e:
            self.log_test("Input Validation", False, f"Error: {str(e)}")
    
    def test_security_headers(self):
        """Test security headers."""
        print("\n🔒 Testing Security Headers...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            headers = response.headers
            
            # Check for security headers
            expected_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options', 
                'X-XSS-Protection',
                'Referrer-Policy'
            ]
            
            missing_headers = []
            for header in expected_headers:
                if header not in headers:
                    missing_headers.append(header)
            
            if not missing_headers:
                self.log_test("Security Headers", True, "All security headers present")
            else:
                self.log_test("Security Headers", False, f"Missing headers: {missing_headers}")
        except Exception as e:
            self.log_test("Security Headers", False, f"Error: {str(e)}")
    
    def test_error_handling(self):
        """Test error handling and information disclosure."""
        print("\n🚨 Testing Error Handling...")
        
        # Test 404 error
        try:
            response = self.session.get(f"{self.base_url}/nonexistent-endpoint")
            
            if response.status_code == 404:
                # Check that error doesn't reveal too much information
                error_text = response.text.lower()
                if "traceback" not in error_text and "stack" not in error_text:
                    self.log_test("Error Information Disclosure", True, "Minimal error information disclosed")
                else:
                    self.log_test("Error Information Disclosure", False, "Too much error information disclosed")
            else:
                self.log_test("Error Information Disclosure", False, f"Expected 404, got {response.status_code}")
        except Exception as e:
            self.log_test("Error Information Disclosure", False, f"Error: {str(e)}")
    
    def test_server_headers(self):
        """Test server header disclosure."""
        print("\n🔍 Testing Server Header Disclosure...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            headers = response.headers
            
            # Check for server header disclosure
            server_header = headers.get('Server', '')
            if 'uvicorn' not in server_header.lower() and 'fastapi' not in server_header.lower():
                self.log_test("Server Header Disclosure", True, "Server information not disclosed")
            else:
                self.log_test("Server Header Disclosure", False, f"Server header present: {server_header}")
        except Exception as e:
            self.log_test("Server Header Disclosure", False, f"Error: {str(e)}")
    
    def test_health_endpoint(self):
        """Test health endpoint functionality."""
        print("\n❤️  Testing Health Endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                if "security" in data and "timestamp" in data:
                    self.log_test("Health Endpoint", True, "Health endpoint working with security info")
                else:
                    self.log_test("Health Endpoint", False, "Health endpoint missing security information")
            else:
                self.log_test("Health Endpoint", False, f"Expected 200, got {response.status_code}")
        except Exception as e:
            self.log_test("Health Endpoint", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all security tests."""
        print("🛡️  Starting Security Validation Tests...")
        print(f"Target: {self.base_url}")
        print("=" * 60)
        
        # Run tests
        self.test_health_endpoint()
        self.test_cors_security()
        self.test_authentication()
        self.test_rate_limiting()
        self.test_input_validation()
        self.test_security_headers()
        self.test_error_handling()
        self.test_server_headers()
        
        # Summary
        print("\n" + "=" * 60)
        print("🛡️  Security Test Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        # Show failed tests
        failed_tests = [result for result in self.test_results if not result["passed"]]
        if failed_tests:
            print("\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        # Save results
        self.save_results()
        
        return passed == total
    
    def save_results(self):
        """Save test results to file."""
        results_file = "security_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "target": self.base_url,
                "summary": {
                    "total_tests": len(self.test_results),
                    "passed": sum(1 for r in self.test_results if r["passed"]),
                    "failed": sum(1 for r in self.test_results if not r["passed"])
                },
                "results": self.test_results
            }, f, indent=2)
        
        print(f"\n📊 Results saved to {results_file}")

def main():
    """Main function to run security tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security validation tests")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the API server")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if server is running
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server not responding correctly at {args.url}")
            print("Please ensure the server is running before running security tests.")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"❌ Cannot connect to server at {args.url}")
        print("Please ensure the server is running before running security tests.")
        sys.exit(1)
    
    # Run tests
    tester = SecurityTester(args.url)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All security tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some security tests failed. Please review the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
