#!/bin/bash

# AI-Powered API Testing Framework - Production Deployment Script
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check .env file
    if [ ! -f .env ]; then
        log_warning ".env file not found. Creating from production template..."
        cp .env.production .env
        log_warning "Please edit .env file with your production values before continuing."
        read -p "Press Enter to continue after editing .env..."
    fi
    
    log_success "Prerequisites check completed."
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build backend image
    log_info "Building backend image..."
    docker build -t api-testing-framework:latest .
    
    # Build frontend image
    log_info "Building frontend image..."
    cd frontend
    docker build -f Dockerfile.prod -t api-testing-frontend:latest .
    cd ..
    
    log_success "Docker images built successfully."
}

# Setup infrastructure
setup_infrastructure() {
    log_info "Setting up infrastructure..."
    
    # Create necessary directories
    mkdir -p data/chromadb
    mkdir -p data/rl_models
    mkdir -p logs
    mkdir -p logs/tensorboard
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    # Set proper permissions
    chmod -R 755 data
    chmod -R 755 logs
    
    log_success "Infrastructure setup completed."
}

# Initialize database
init_database() {
    log_info "Initializing database..."
    
    # Start database services
    docker-compose -f docker-compose.prod.yml up -d postgres redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    sleep 30
    
    # Run database migrations
    docker-compose -f docker-compose.prod.yml run --rm api migrate
    
    log_success "Database initialized successfully."
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    # Deploy with production configuration
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 60
    
    # Health check
    health_check
    
    log_success "Services deployed successfully."
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    # Check API health
    for i in {1..10}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API is healthy."
            break
        elif [ $i -eq 10 ]; then
            log_error "API health check failed after 10 attempts."
            return 1
        else
            log_info "API not ready, waiting... (attempt $i/10)"
            sleep 10
        fi
    done
    
    # Check frontend
    if curl -f http://localhost &> /dev/null; then
        log_success "Frontend is healthy."
    else
        log_warning "Frontend health check failed. Check nginx configuration."
    fi
    
    # Check database
    if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U postgres -d api_testing &> /dev/null; then
        log_success "Database is healthy."
    else
        log_error "Database health check failed."
        return 1
    fi
    
    # Check Redis
    if docker-compose -f docker-compose.prod.yml exec redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis is healthy."
    else
        log_error "Redis health check failed."
        return 1
    fi
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring configurations
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-testing-framework'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

    # Create Grafana datasource configuration
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log_success "Monitoring setup completed."
}

# Backup database
backup_database() {
    log_info "Creating database backup..."
    
    BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
    
    docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U postgres api_testing > "backups/$BACKUP_FILE"
    
    log_success "Database backup created: backups/$BACKUP_FILE"
}

# Rollback deployment
rollback() {
    log_warning "Rolling back deployment..."
    
    # Stop current services
    docker-compose -f docker-compose.prod.yml down
    
    # Restore from backup if available
    LATEST_BACKUP=$(ls -t backups/*.sql 2>/dev/null | head -n1)
    if [ -n "$LATEST_BACKUP" ]; then
        log_info "Restoring from backup: $LATEST_BACKUP"
        docker-compose -f docker-compose.prod.yml up -d postgres
        sleep 30
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U postgres api_testing < "$LATEST_BACKUP"
    fi
    
    log_warning "Rollback completed. Please investigate the issue before redeploying."
}

# Cleanup old images and containers
cleanup() {
    log_info "Cleaning up old Docker resources..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log_success "Cleanup completed."
}

# Show deployment status
status() {
    log_info "Deployment status:"
    
    echo ""
    echo "Services:"
    docker-compose -f docker-compose.prod.yml ps
    
    echo ""
    echo "Health Checks:"
    
    # API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "API: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "API: ${RED}✗ Unhealthy${NC}"
    fi
    
    # Frontend health
    if curl -f http://localhost &> /dev/null; then
        echo -e "Frontend: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "Frontend: ${RED}✗ Unhealthy${NC}"
    fi
    
    # Database health
    if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U postgres -d api_testing &> /dev/null; then
        echo -e "Database: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "Database: ${RED}✗ Unhealthy${NC}"
    fi
    
    # Redis health
    if docker-compose -f docker-compose.prod.yml exec redis redis-cli ping | grep -q "PONG" 2> /dev/null; then
        echo -e "Redis: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "Redis: ${RED}✗ Unhealthy${NC}"
    fi
    
    echo ""
    echo "Access URLs:"
    echo "  - API: http://localhost:8000"
    echo "  - Frontend: http://localhost"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3001"
}

# Main deployment function
deploy() {
    log_info "Starting production deployment..."
    
    # Create backup directory
    mkdir -p backups
    
    # Run deployment steps
    check_prerequisites
    setup_infrastructure
    setup_monitoring
    build_images
    
    # Backup existing database if it exists
    if docker-compose -f docker-compose.prod.yml ps postgres | grep -q "Up"; then
        backup_database
    fi
    
    init_database
    deploy_services
    
    log_success "Production deployment completed successfully!"
    echo ""
    status
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "status")
        status
        ;;
    "health-check")
        health_check
        ;;
    "backup")
        backup_database
        ;;
    "rollback")
        rollback
        ;;
    "cleanup")
        cleanup
        ;;
    "build")
        build_images
        ;;
    *)
        echo "Usage: $0 {deploy|status|health-check|backup|rollback|cleanup|build}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Full production deployment"
        echo "  status       - Show current deployment status"
        echo "  health-check - Run health checks on all services"
        echo "  backup       - Create database backup"
        echo "  rollback     - Rollback to previous deployment"
        echo "  cleanup      - Clean up unused Docker resources"
        echo "  build        - Build Docker images only"
        exit 1
        ;;
esac
