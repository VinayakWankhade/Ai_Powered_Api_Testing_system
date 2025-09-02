"""
Quick startup test for the AI-Powered API Testing Framework.
"""

import sys
import os

# Set environment variables
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_startup.db")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "ERROR")  # Reduce noise

def test_imports():
    """Test that we can import core modules."""
    print("Testing core imports...")
    
    try:
        sys.path.insert(0, "src")
        
        # Test database imports
        from database.connection import create_tables
        print("✅ Database connection module")
        
        # Test database models
        from database.models import Base
        print("✅ Database models")
        
        # Create tables
        create_tables()
        print("✅ Database tables created")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test HTTP client functionality
        import httpx
        import asyncio
        
        async def test_http():
            async with httpx.AsyncClient() as client:
                response = await client.get("https://httpbin.org/get", timeout=10)
                return response.status_code == 200
        
        success = asyncio.run(test_http())
        if success:
            print("✅ HTTP client working")
        else:
            print("❌ HTTP client failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AI-Powered API Testing Framework - Startup Test")
    print("=" * 55)
    
    # Run tests
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 55)
    if imports_ok and functionality_ok:
        print("🎉 STARTUP TEST PASSED!")
        print("✅ Core system is ready to run")
        print("\nTo start the system:")
        print("1. docker-compose up -d")
        print("2. Or: python -m uvicorn src.api.main:app --reload")
    else:
        print("⚠️  STARTUP TEST FAILED!")
        print("Please check the errors above")
    
    print("=" * 55)
