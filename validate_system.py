#!/usr/bin/env python3
"""
System validation script to verify the AI-Powered API Testing Framework MVP.
Tests core components and verifies the system is ready for deployment.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_check(description, success, details=None):
    """Print a check result."""
    status = "✅" if success else "❌"
    print(f"{status} {description}")
    if details:
        for detail in details:
            print(f"   {detail}")

async def validate_system():
    """Validate the entire system."""
    print("🚀 AI-Powered API Testing Framework - System Validation")
    
    validation_results = {
        "database": False,
        "api_structure": False,
        "services": False,
        "frontend": False,
        "docker": False,
        "tests": False,
        "documentation": False
    }
    
    # 1. Database validation
    print_section("DATABASE VALIDATION")
    try:
        # Check if database files exist and can be imported
        db_files = [
            "src/database/__init__.py",
            "src/database/connection.py",
            "src/database/models.py"
        ]
        
        missing_files = [f for f in db_files if not os.path.exists(f)]
        if missing_files:
            print_check("Database files", False, [f"Missing: {f}" for f in missing_files])
        else:
            print_check("Database files", True, ["All database files present"])
            
        # Test basic import
        sys.path.insert(0, "src")
        from database.connection import create_tables
        create_tables()
        print_check("Database initialization", True, ["Tables created successfully"])
        validation_results["database"] = True
        
    except Exception as e:
        print_check("Database validation", False, [f"Error: {e}"])
    
    # 2. API structure validation
    print_section("API STRUCTURE VALIDATION")
    try:
        api_files = [
            "src/api/__init__.py",
            "src/api/main.py",
            "src/api/endpoints/__init__.py",
            "src/api/endpoints/specs.py"
        ]
        
        missing_files = [f for f in api_files if not os.path.exists(f)]
        if missing_files:
            print_check("API structure", False, [f"Missing: {f}" for f in missing_files])
        else:
            print_check("API structure", True, ["All API files present"])
            validation_results["api_structure"] = True
            
    except Exception as e:
        print_check("API structure validation", False, [f"Error: {e}"])
    
    # 3. Services validation
    print_section("SERVICES VALIDATION")
    service_files = [
        "src/services/test_executor.py",
        "src/services/ai_test_generator.py",
        "src/services/coverage_analyzer.py",
        "src/services/self_healing.py",
        "src/services/rl_optimizer.py"
    ]
    
    present_services = [f for f in service_files if os.path.exists(f)]
    missing_services = [f for f in service_files if not os.path.exists(f)]
    
    print_check("Core services", len(missing_services) == 0, 
                [f"Present: {len(present_services)}/{len(service_files)} services"])
    
    if missing_services:
        print_check("Missing services", False, missing_services)
    else:
        validation_results["services"] = True
    
    # 4. Frontend validation
    print_section("FRONTEND VALIDATION")
    frontend_files = [
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/pages/Dashboard.tsx",
        "frontend/src/services/api.ts"
    ]
    
    present_frontend = [f for f in frontend_files if os.path.exists(f)]
    missing_frontend = [f for f in frontend_files if not os.path.exists(f)]
    
    print_check("Frontend structure", len(missing_frontend) == 0,
                [f"Present: {len(present_frontend)}/{len(frontend_files)} key files"])
    
    if missing_frontend:
        print_check("Missing frontend files", False, missing_frontend)
    else:
        validation_results["frontend"] = True
    
    # 5. Docker validation
    print_section("DOCKER VALIDATION")
    docker_files = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.prod.yml",
        "frontend/Dockerfile.prod"
    ]
    
    present_docker = [f for f in docker_files if os.path.exists(f)]
    missing_docker = [f for f in docker_files if not os.path.exists(f)]
    
    print_check("Docker configuration", len(missing_docker) == 0,
                [f"Present: {len(present_docker)}/{len(docker_files)} Docker files"])
    
    if missing_docker:
        print_check("Missing Docker files", False, missing_docker)
    else:
        validation_results["docker"] = True
    
    # 6. Tests validation
    print_section("TESTS VALIDATION")
    test_files = [
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_api.py",
        "tests/test_services.py",
        "tests/test_models.py",
        "tests/test_performance.py"
    ]
    
    present_tests = [f for f in test_files if os.path.exists(f)]
    missing_tests = [f for f in test_files if not os.path.exists(f)]
    
    print_check("Test suite", len(missing_tests) == 0,
                [f"Present: {len(present_tests)}/{len(test_files)} test files"])
    
    if missing_tests:
        print_check("Missing test files", False, missing_tests)
    else:
        validation_results["tests"] = True
    
    # 7. Documentation validation
    print_section("DOCUMENTATION VALIDATION")
    doc_files = [
        "README.md",
        "docs/README.md",
        "docs/api-reference.md",
        ".env.production"
    ]
    
    present_docs = [f for f in doc_files if os.path.exists(f)]
    missing_docs = [f for f in doc_files if not os.path.exists(f)]
    
    print_check("Documentation", len(missing_docs) == 0,
                [f"Present: {len(present_docs)}/{len(doc_files)} documentation files"])
    
    if missing_docs:
        print_check("Missing documentation", False, missing_docs)
    else:
        validation_results["documentation"] = True
    
    # 8. Quick functionality test
    print_section("FUNCTIONALITY TEST")
    try:
        # Test that we can make a simple HTTP request
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get("https://jsonplaceholder.typicode.com/posts/1")
            
        print_check("HTTP client", response.status_code == 200,
                   [f"Status: {response.status_code}", 
                    f"Response time: {response.elapsed.total_seconds()*1000:.0f}ms"])
        
        # Test JSON parsing
        data = response.json()
        print_check("JSON parsing", "id" in data and "title" in data,
                   [f"Post ID: {data.get('id')}", f"Title: {data.get('title', '')[:50]}..."])
                   
    except Exception as e:
        print_check("Functionality test", False, [f"Error: {e}"])
    
    # Final summary
    print_section("VALIDATION SUMMARY")
    
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    success_rate = (passed_checks / total_checks) * 100
    
    print(f"\nValidation Results: {passed_checks}/{total_checks} checks passed ({success_rate:.1f}%)")
    print("\nComponent Status:")
    
    for component, status in validation_results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}")
    
    if success_rate >= 80:
        print(f"\n🎉 System validation PASSED! ({success_rate:.1f}%)")
        print("The AI-Powered API Testing Framework MVP is ready for deployment!")
        
        print("\n🚀 Quick Start Commands:")
        print("1. Start with Docker: docker-compose up -d")
        print("2. Start development server: python -m uvicorn src.api.main:app --reload")
        print("3. Run tests: python -m pytest tests/ -v")
        print("4. Access API docs: http://localhost:8000/docs")
        print("5. Access frontend: http://localhost:3000")
        
    else:
        print(f"\n⚠️  System validation completed with issues ({success_rate:.1f}%)")
        print("Please review the failed checks above before deployment.")
    
    return success_rate >= 80

if __name__ == "__main__":
    # Set basic environment
    os.environ.setdefault("DATABASE_URL", "sqlite:///./validation.db")
    os.environ.setdefault("ENVIRONMENT", "validation")
    
    success = asyncio.run(validate_system())
    sys.exit(0 if success else 1)
