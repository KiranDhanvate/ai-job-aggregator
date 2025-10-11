# 🚀 AI Job Aggregator & Recommender API

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time job scraping meets AI-powered recommendations** - A comprehensive solution for intelligent job discovery and personalized career guidance.

## 🌟 Features

### 🔍 **Real-Time Job Scraping**
- **Multi-platform scraping** from LinkedIn, Indeed, Glassdoor, Naukri, and more
- **Live data fetching** - Always up-to-date job postings
- **Full job descriptions** with detailed company information
- **Advanced filtering** by location, remote work, job type, and recency
- **Rate limiting protection** with intelligent retry mechanisms

### 🤖 **AI-Powered Recommendations**
- **Personalized job matching** using machine learning
- **Skill-based recommendations** based on job requirements
- **Career path suggestions** with growth opportunities
- **Salary predictions** and market insights
- **Company culture analysis** and fit scoring

### 🚀 **Production-Ready API**
- **FastAPI-based** RESTful API with automatic documentation
- **Async processing** for high-performance scraping
- **Comprehensive error handling** and logging
- **Health monitoring** and performance metrics
- **CORS enabled** for frontend integration

### 📊 **Data Intelligence**
- **Real-time analytics** and scraping statistics
- **Data quality validation** and completeness reports
- **Export capabilities** (CSV, JSON formats)
- **Database integration** with SQLAlchemy
- **Performance monitoring** and optimization insights

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Python 3.10+, Uvicorn |
| **ML/AI** | TensorFlow, Scikit-learn, Pandas |
| **Scraping** | JobSpy, BeautifulSoup4, Requests |
| **Database** | SQLAlchemy, SQLite/PostgreSQL |
| **DevOps** | Poetry, Pre-commit, Pytest |
| **API** | RESTful APIs, Async processing |

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KiranDhanvate/ai-job-aggregator.git
   cd ai-job-aggregator
   ```

2. **Install dependencies**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install project dependencies
   poetry install
   
   # Activate virtual environment
   poetry shell
   ```

3. **Train the recommendation model**
   ```bash
   python -m recommendation_model.train
   ```

4. **Start the API server**
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Access the API documentation**
   - Open your browser to: `http://localhost:8000/docs`
   - Interactive API documentation with Swagger UI

## 📖 API Usage

### 🔍 **Basic Job Scraping**

```bash
# Quick job search
curl -X GET "http://localhost:8000/scrape?search_term=python%20developer&location=San%20Francisco&results_wanted=10"
```

### 🤖 **AI-Powered Recommendations**

```bash
# Get personalized job recommendations
curl -X POST "http://localhost:8000/scrape-and-recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "search_term": "machine learning engineer",
    "location": "Remote",
    "user_skills": ["Python", "TensorFlow", "AWS", "Docker"],
    "experience_years": 3,
    "preferred_salary_min": 80000
  }'
```

### 📊 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check and metrics |
| `/scrape` | GET/POST | Job scraping with various filters |
| `/scrape-and-recommend` | POST | AI-powered job recommendations |
| `/stats` | GET | Scraping statistics and analytics |
| `/docs` | GET | Interactive API documentation |

## 🔧 Configuration

### Environment Variables
Create a `.env` file in your project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Scraping Configuration
MAX_RESULTS_PER_REQUEST=50
REQUEST_TIMEOUT=30
RATE_LIMIT_DELAY=1

# Database Configuration
DATABASE_URL=sqlite:///./jobspy.db

# ML Model Configuration
MODEL_PATH=./artifacts/recommendation_model.h5
```

### Supported Job Sites
- **LinkedIn** - Professional networking and job postings
- **Indeed** - Global job search platform
- **Glassdoor** - Company reviews and job listings
- **Naukri** - Indian job portal
- **BDJobs** - Bangladesh job portal
- **ZipRecruiter** - US job search platform

## 📊 Example Usage

### Python Client Example

```python
import requests
import json

# Basic job scraping
response = requests.post('http://localhost:8000/scrape', json={
    'search_term': 'data scientist',
    'location': 'New York',
    'results_wanted': 20,
    'is_remote': True,
    'hours_old': 72
})

jobs = response.json()
print(f"Found {jobs['count']} jobs")

# AI-powered recommendations
recommendations = requests.post('http://localhost:8000/scrape-and-recommend', json={
    'search_term': 'software engineer',
    'location': 'Remote',
    'user_skills': ['Python', 'React', 'AWS'],
    'experience_years': 2,
    'preferred_salary_min': 70000
})

recommended_jobs = recommendations.json()
print(f"AI found {recommended_jobs['count']} matching jobs")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

// Scrape jobs
async function scrapeJobs() {
    try {
        const response = await axios.post('http://localhost:8000/scrape', {
            search_term: 'frontend developer',
            location: 'San Francisco',
            results_wanted: 15
        });
        
        console.log(`Found ${response.data.count} jobs`);
        return response.data.jobs;
    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Get AI recommendations
async function getRecommendations() {
    try {
        const response = await axios.post('http://localhost:8000/scrape-and-recommend', {
            search_term: 'full stack developer',
            user_skills: ['JavaScript', 'Python', 'React', 'Node.js'],
            experience_years: 3,
            preferred_salary_min: 80000
        });
        
        return response.data.recommended_jobs;
    } catch (error) {
        console.error('Error:', error.message);
    }
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=jobspy --cov=api_server

# Run specific test file
pytest tests/test_scraper.py -v
```

## 📈 Performance & Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Statistics
```bash
curl http://localhost:8000/stats
```

### Metrics Dashboard
The API provides real-time metrics including:
- Total requests processed
- Jobs scraped per site
- Average response times
- Error rates and success rates
- Model prediction accuracy

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment (AWS/GCP/Azure)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_HOST=0.0.0.0
export API_PORT=8000

# Run with Gunicorn for production
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run linting
pre-commit run --all-files
```

## 📋 Roadmap

### 🎯 **Upcoming Features**
- [ ] **Advanced ML Models** - Deep learning for better recommendations
- [ ] **Real-time Notifications** - WebSocket support for live updates
- [ ] **Company Analytics** - Detailed company insights and trends
- [ ] **Mobile App** - React Native mobile application
- [ ] **Chrome Extension** - Browser extension for job tracking
- [ ] **Salary Predictions** - ML-powered salary estimation
- [ ] **Skills Gap Analysis** - Personalized learning recommendations

### 🔮 **Future Enhancements**
- [ ] **Multi-language Support** - Support for multiple job markets
- [ ] **Advanced Filtering** - Industry-specific filters and criteria
- [ ] **API Rate Limiting** - Intelligent rate limiting and caching
- [ ] **Data Export** - Advanced export options and integrations
- [ ] **Analytics Dashboard** - Comprehensive analytics and reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kiran Dhanvate**
- GitHub: [@KiranDhanvate](https://github.com/KiranDhanvate)
- LinkedIn: [Kiran Dhanvate](https://linkedin.com/in/kiran-dhanvate)
- Portfolio: [Your Portfolio Website]

## 🙏 Acknowledgments

- **JobSpy** - Core scraping functionality
- **FastAPI** - High-performance API framework
- **TensorFlow** - Machine learning capabilities
- **Open Source Community** - For continuous improvements

## 📞 Support

- **Documentation**: [Full API Documentation](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/KiranDhanvate/ai-job-aggregator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KiranDhanvate/ai-job-aggregator/discussions)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Kiran Dhanvate](https://github.com/KiranDhanvate)

</div>