# AI-Powered API Testing System - Production Dockerfile

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p data logs data/chromadb data/rl_models logs/tensorboard && \
    chown -R appuser:appuser /app

# Create production entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# Initialize database if needed
if [ "$INIT_DB" = "true" ]; then
    echo "Initializing database..."
    python -c "from src.database.connection import create_tables; create_tables()"
fi

# Start the appropriate service
case "$1" in
    "api")
        echo "Starting API server..."
        if [ "$ENVIRONMENT" = "production" ]; then
            exec gunicorn src.api.main:app \
                --bind 0.0.0.0:8000 \
                --workers ${WORKERS:-4} \
                --worker-class uvicorn.workers.UvicornWorker \
                --timeout ${TIMEOUT:-300} \
                --keep-alive 2 \
                --max-requests 1000 \
                --max-requests-jitter 100 \
                --preload
        else
            exec uvicorn src.api.main:app \
                --host 0.0.0.0 \
                --port 8000 \
                --reload
        fi
        ;;
    "worker")
        echo "Starting Celery worker..."
        exec celery -A src.tasks worker \
            --loglevel=info \
            --concurrency=${WORKER_CONCURRENCY:-4}
        ;;
    "scheduler")
        echo "Starting Celery beat scheduler..."
        exec celery -A src.tasks beat \
            --loglevel=info
        ;;
    "migrate")
        echo "Running database migrations..."
        python -c "from src.database.connection import create_tables; create_tables()"
        ;;
    "test")
        echo "Running tests..."
        exec pytest tests/ -v
        ;;
    *)
        exec "$@"
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
