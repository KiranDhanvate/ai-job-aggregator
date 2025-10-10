# 🚀 Deployment Guide

This guide covers various deployment options for the AI Job Aggregator & Recommender API.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Configuration](#production-configuration)
- [Monitoring & Logging](#monitoring--logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 2GB RAM minimum (4GB recommended)
- **Storage**: 10GB free space
- **Network**: Stable internet connection for scraping

### Software Dependencies
- **Poetry**: For dependency management
- **Docker**: For containerized deployment (optional)
- **Git**: For version control

## 🏠 Local Development

### Quick Start
```bash
# Clone the repository
git clone https://github.com/KiranDhanvate/ai-job-aggregator.git
cd ai-job-aggregator

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Train the ML model
python -m recommendation_model.train

# Start the development server
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

### Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Test basic functionality
curl http://localhost:8000/scrape?search_term=python&results_wanted=5
```

## 🐳 Docker Deployment

### Single Container
```bash
# Build the image
docker build -t ai-job-aggregator .

# Run the container
docker run -d \
  --name job-aggregator \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ai-job-aggregator
```

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Compose Services
- **API Server**: Main application
- **Redis**: Caching layer
- **PostgreSQL**: Database
- **Nginx**: Reverse proxy
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboard

## ☁️ Cloud Deployment

### AWS Deployment

#### Using EC2
```bash
# Launch EC2 instance (Ubuntu 20.04)
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose

# Clone and deploy
git clone https://github.com/KiranDhanvate/ai-job-aggregator.git
cd ai-job-aggregator
docker-compose up -d
```

#### Using AWS ECS
```yaml
# ecs-task-definition.json
{
  "family": "ai-job-aggregator",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "job-aggregator",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ai-job-aggregator:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "API_PORT",
          "value": "8000"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-job-aggregator",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Using AWS Lambda (Serverless)
```python
# lambda_handler.py
import json
from mangum import Mangum
from api_server import app

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Platform

#### Using Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/ai-job-aggregator
gcloud run deploy --image gcr.io/PROJECT-ID/ai-job-aggregator \
  --platform managed --region us-central1 --allow-unauthenticated
```

#### Using App Engine
```yaml
# app.yaml
runtime: python39
service: job-aggregator

env_variables:
  API_HOST: 0.0.0.0
  API_PORT: 8080

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6

handlers:
- url: /.*
  script: auto
```

### Microsoft Azure

#### Using Container Instances
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name job-aggregator \
  --image your-registry.azurecr.io/ai-job-aggregator:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables API_HOST=0.0.0.0 API_PORT=8000
```

#### Using App Service
```bash
# Deploy to Azure App Service
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name my-job-aggregator \
  --deployment-container-image-name your-registry.azurecr.io/ai-job-aggregator:latest
```

## ⚙️ Production Configuration

### Environment Variables
```bash
# Production settings
export ENVIRONMENT=production
export DEBUG=False
export WORKERS=4
export API_HOST=0.0.0.0
export API_PORT=8000

# Database
export DATABASE_URL=postgresql://user:password@localhost:5432/jobspy

# Security
export SECRET_KEY=your-secure-secret-key
export CORS_ORIGINS=["https://yourdomain.com"]

# Monitoring
export LOG_LEVEL=INFO
export PROMETHEUS_ENABLED=True
```

### Nginx Configuration
```nginx
# /etc/nginx/sites-available/job-aggregator
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

### SSL/TLS Setup
```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Process Management
```bash
# Using systemd
sudo nano /etc/systemd/system/job-aggregator.service
```

```ini
[Unit]
Description=AI Job Aggregator API
After=network.target

[Service]
Type=exec
User=www-data
WorkingDirectory=/opt/job-aggregator
Environment=PATH=/opt/job-aggregator/.venv/bin
ExecStart=/opt/job-aggregator/.venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable job-aggregator
sudo systemctl start job-aggregator
sudo systemctl status job-aggregator
```

## 📊 Monitoring & Logging

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'job-aggregator'
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "AI Job Aggregator Metrics",
    "panels": [
      {
        "title": "API Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Log Management
```bash
# Using logrotate
sudo nano /etc/logrotate.d/job-aggregator
```

```
/opt/job-aggregator/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload job-aggregator
    endscript
}
```

### Health Checks
```bash
# Custom health check script
#!/bin/bash
# health-check.sh

API_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ API is healthy"
    exit 0
else
    echo "❌ API is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

## 🔒 Security Considerations

### Network Security
```bash
# Firewall configuration
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### Application Security
```python
# Security middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)
```

### Rate Limiting
```python
# Rate limiting implementation
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/scrape")
@limiter.limit("10/minute")
async def scrape_jobs(request: Request, ...):
    # Implementation
```

### Data Protection
```python
# Environment-based configuration
import os
from cryptography.fernet import Fernet

# Encrypt sensitive data
def encrypt_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 PID

# Or use different port
uvicorn api_server:app --port 8001
```

#### 2. Memory Issues
```bash
# Check memory usage
free -h

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Database Connection Issues
```bash
# Check database status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U username -d jobspy -c "SELECT 1;"
```

#### 4. Scraping Failures
```bash
# Check network connectivity
curl -I https://www.linkedin.com
curl -I https://www.indeed.com

# Test with verbose logging
uvicorn api_server:app --log-level debug
```

### Performance Optimization

#### 1. Increase Worker Processes
```bash
# For CPU-bound workloads
uvicorn api_server:app --workers 4

# For I/O-bound workloads
uvicorn api_server:app --workers 8 --worker-class uvicorn.workers.UvicornWorker
```

#### 2. Enable Caching
```python
# Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

#### 3. Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_jobs_company ON jobs(company);
CREATE INDEX idx_jobs_location ON jobs(location);
CREATE INDEX idx_jobs_date_posted ON jobs(date_posted);
CREATE INDEX idx_jobs_site ON jobs(site);
```

### Monitoring Commands
```bash
# Check API status
curl http://localhost:8000/health

# View logs
tail -f logs/app.log

# Monitor resource usage
htop

# Check disk space
df -h

# Monitor network connections
netstat -tulpn
```

## 📞 Support

### Getting Help
- **Documentation**: Check the README.md and API docs
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions
- **Email**: Contact the maintainers directly

### Emergency Contacts
- **Critical Issues**: Create GitHub issue with "urgent" label
- **Security Issues**: Email security@yourdomain.com
- **Performance Issues**: Check monitoring dashboards first

---

*This deployment guide is regularly updated. Please check for the latest version.*
