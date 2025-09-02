#!/usr/bin/env python3
"""
Startup script for the AI-Powered API Testing System.
Provides easy initialization and management of the system.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
app = typer.Typer(help="AI-Powered API Testing System Management")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        console.print("[bold red]Error: Python 3.9+ is required[/bold red]")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "fastapi", "uvicorn", "sqlalchemy", "pydantic", 
        "httpx", "requests", "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        console.print(f"[bold red]Missing required packages: {', '.join(missing_packages)}[/bold red]")
        console.print("Run: [bold cyan]pip install -r requirements.txt[/bold cyan]")
        return False
    
    return True

def setup_environment():
    """Set up environment variables and directories."""
    
    # Create necessary directories
    directories = ["data", "logs", "data/chromadb", "data/rl_models", "logs/tensorboard"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        console.print(f"✓ Created directory: {directory}")
    
    # Check if .env file exists
    if not Path(".env").exists():
        if Path(".env.example").exists():
            console.print("Creating .env file from .env.example...")
            import shutil
            shutil.copy(".env.example", ".env")
            console.print("[yellow]Please edit .env file with your configuration before starting the system[/yellow]")
        else:
            console.print("[yellow]No .env file found. Creating a basic one...[/yellow]")
            create_basic_env_file()
    
    # Check critical environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        console.print("[yellow]Warning: OPENAI_API_KEY not configured. AI features will be limited.[/yellow]")

def create_basic_env_file():
    """Create a basic .env file with default values."""
    
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Database Configuration
DATABASE_URL=sqlite:///./data/api_testing.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/api_testing.log

# RAG Configuration
CHROMADB_PATH=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RL Configuration
RL_MODEL_PATH=./data/rl_models
TENSORBOARD_LOG_DIR=./logs/tensorboard

# Testing Configuration
TEST_TIMEOUT=300
MAX_CONCURRENT_TESTS=5
SANDBOX_MODE=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    console.print("✓ Created basic .env file")

def initialize_database():
    """Initialize the database with tables."""
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing database...", total=None)
            
            from src.database.connection import create_tables
            create_tables()
            
            progress.update(task, description="Database initialized successfully")
        
        console.print("✓ Database tables created")
        return True
        
    except Exception as e:
        console.print(f"[bold red]Failed to initialize database: {str(e)}[/bold red]")
        return False

def start_backend_service(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    workers: int = 1
):
    """Start the FastAPI backend service."""
    
    try:
        console.print(f"Starting API server on {host}:{port}...")
        
        if reload:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ]
        else:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", host,
                "--port", str(port),
                "--workers", str(workers)
            ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]API server stopped by user[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to start API server: {str(e)}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error starting API server: {str(e)}[/bold red]")

def start_frontend_service():
    """Start the React frontend development server."""
    
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        console.print("[bold red]Frontend directory not found[/bold red]")
        return False
    
    try:
        console.print("Starting frontend development server...")
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            console.print("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        # Start development server
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, check=True)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Frontend server stopped by user[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Failed to start frontend server: {str(e)}[/bold red]")
    except FileNotFoundError:
        console.print("[bold red]Node.js/npm not found. Please install Node.js first.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error starting frontend: {str(e)}[/bold red]")

@app.command()
def setup():
    """Set up the system for first-time use."""
    
    console.print(Panel.fit("🚀 AI-Powered API Testing System Setup", style="bold blue"))
    
    # Check prerequisites
    console.print("\n📋 Checking prerequisites...")
    check_python_version()
    
    if not check_dependencies():
        console.print("\n[bold red]Please install dependencies first:[/bold red]")
        console.print("[bold cyan]pip install -r requirements.txt[/bold cyan]")
        sys.exit(1)
    
    # Setup environment
    console.print("\n🔧 Setting up environment...")
    setup_environment()
    
    # Initialize database
    console.print("\n🗄️ Initializing database...")
    if not initialize_database():
        sys.exit(1)
    
    console.print("\n✅ [bold green]Setup completed successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Edit .env file with your OpenAI API key")
    console.print("2. Run: [bold cyan]python start_system.py dev[/bold cyan]")

@app.command()
def dev():
    """Start the system in development mode."""
    
    console.print(Panel.fit("🚀 Starting Development Environment", style="bold green"))
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if system is set up
    if not Path("data").exists() or not Path(".env").exists():
        console.print("[bold yellow]System not set up. Running setup first...[/bold yellow]")
        setup()
    
    # Start backend
    console.print("\n🔧 Starting backend API server...")
    start_backend_service(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True
    )

@app.command()
def frontend():
    """Start only the frontend development server."""
    
    console.print(Panel.fit("🎨 Starting Frontend Development Server", style="bold cyan"))
    start_frontend_service()

@app.command()
def production(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(4, help="Number of worker processes")
):
    """Start the system in production mode."""
    
    console.print(Panel.fit("🌟 Starting Production Environment", style="bold magenta"))
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Start backend in production mode
    console.print(f"\n🚀 Starting production API server with {workers} workers...")
    start_backend_service(
        host=host,
        port=port,
        reload=False,
        workers=workers
    )

@app.command()
def status():
    """Check system status and health."""
    
    console.print(Panel.fit("📊 System Status Check", style="bold yellow"))
    
    # Check if .env exists
    env_status = "✅ Found" if Path(".env").exists() else "❌ Missing"
    console.print(f"Environment file: {env_status}")
    
    # Check if data directory exists
    data_status = "✅ Found" if Path("data").exists() else "❌ Missing"
    console.print(f"Data directory: {data_status}")
    
    # Check database file
    db_status = "✅ Found" if Path("data/api_testing.db").exists() else "❌ Missing"
    console.print(f"Database file: {db_status}")
    
    # Check dependencies
    deps_status = "✅ Satisfied" if check_dependencies() else "❌ Missing packages"
    console.print(f"Dependencies: {deps_status}")
    
    # Try to connect to API server
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        api_status = "✅ Running" if response.status_code == 200 else f"❌ Error {response.status_code}"
    except requests.exceptions.ConnectionError:
        api_status = "❌ Not running"
    except Exception as e:
        api_status = f"❌ Error: {str(e)}"
    
    console.print(f"API Server: {api_status}")
    
    # Summary
    console.print("\n📋 Quick Start Commands:")
    console.print("Setup system: [bold cyan]python start_system.py setup[/bold cyan]")
    console.print("Development: [bold cyan]python start_system.py dev[/bold cyan]")
    console.print("Production: [bold cyan]python start_system.py production[/bold cyan]")

@app.command()
def test():
    """Run the test suite."""
    
    console.print(Panel.fit("🧪 Running Test Suite", style="bold green"))
    
    try:
        console.print("Running tests...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short"
        ], check=True)
        
        console.print("✅ [bold green]All tests passed![/bold green]")
        
    except subprocess.CalledProcessError as e:
        console.print(f"❌ [bold red]Tests failed with exit code {e.returncode}[/bold red]")
        sys.exit(e.returncode)
    except FileNotFoundError:
        console.print("[bold red]pytest not found. Install with: pip install pytest[/bold red]")
        sys.exit(1)

@app.command()
def docker():
    """Start the system using Docker Compose."""
    
    console.print(Panel.fit("🐳 Starting with Docker Compose", style="bold blue"))
    
    if not Path("docker-compose.yml").exists():
        console.print("[bold red]docker-compose.yml not found[/bold red]")
        sys.exit(1)
    
    try:
        console.print("Starting services with Docker Compose...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        console.print("✅ [bold green]Services started successfully![/bold green]")
        console.print("\nServices available at:")
        console.print("• API Server: http://localhost:8000")
        console.print("• API Documentation: http://localhost:8000/docs")
        console.print("• Dashboard: http://localhost:8050")
        
    except subprocess.CalledProcessError as e:
        console.print(f"❌ [bold red]Docker Compose failed: {str(e)}[/bold red]")
    except FileNotFoundError:
        console.print("[bold red]Docker Compose not found. Please install Docker first.[/bold red]")

@app.command()
def stop():
    """Stop all services."""
    
    console.print("Stopping services...")
    
    try:
        # Stop Docker services if running
        if Path("docker-compose.yml").exists():
            subprocess.run(["docker-compose", "down"], check=False)
        
        console.print("✅ Services stopped")
        
    except Exception as e:
        console.print(f"Warning: {str(e)}")

def main():
    """Main entry point."""
    
    # Display banner
    console.print("\n" + "="*60)
    console.print("🤖 AI-Powered API Testing System")
    console.print("Advanced API testing with LLMs, RAG, and RL")
    console.print("="*60 + "\n")
    
    # Run typer app
    app()

if __name__ == "__main__":
    main()
