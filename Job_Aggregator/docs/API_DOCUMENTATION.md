# API Documentation

## Overview

The AI Job Aggregator Recommendation API provides RESTful endpoints for job recommendations using the ConvFM machine learning model. The API is built with FastAPI and provides real-time job recommendations, batch processing, and model management capabilities.

## Base URL

```
http://localhost:8001
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement proper authentication mechanisms such as API keys, JWT tokens, or OAuth.

## Content Type

All requests and responses use JSON format with `Content-Type: application/json`.

## Endpoints

### 1. Root Endpoint

**GET** `/`

Get API information and available endpoints.

#### Response
```json
{
  "message": "AI Job Aggregator Recommendation API",
  "version": "1.0.0",
  "status": "healthy",
  "endpoints": {
    "/recommend": "POST - Get job recommendations for a user",
    "/batch_recommend": "POST - Get batch recommendations",
    "/health": "GET - Check API health",
    "/model_info": "GET - Model information",
    "/train": "POST - Train new model",
    "/docs": "GET - API documentation"
  }
}
```

### 2. Health Check

**GET** `/health`

Check API health and model status.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2023-12-07T10:30:00Z",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

### 3. Model Information

**GET** `/model_info`

Get detailed information about the loaded model.

#### Response
```json
{
  "model_type": "ConvFM",
  "version": "1.0.0",
  "num_users": 1000,
  "num_jobs": 5000,
  "num_skills": 2500,
  "loaded_at": "2023-12-07T10:30:00Z"
}
```

#### Error Response
```json
{
  "detail": "Model not loaded"
}
```

**Status Code:** 503 Service Unavailable

### 4. Job Recommendations

**POST** `/recommend`

Get personalized job recommendations for a user.

#### Request Body
```json
{
  "user_id": 123,
  "max_recommendations": 10,
  "location_preference": "San Francisco",
  "company_preference": "Google",
  "salary_min": 100000,
  "job_type": "fulltime",
  "is_remote": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_id` | integer | Yes | - | User identifier |
| `max_recommendations` | integer | No | 10 | Maximum number of recommendations (1-50) |
| `location_preference` | string | No | - | Preferred job location |
| `company_preference` | string | No | - | Preferred company |
| `salary_min` | number | No | - | Minimum salary requirement |
| `job_type` | string | No | - | Job type preference |
| `is_remote` | boolean | No | - | Remote work preference |

#### Response
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "job_id": 456,
      "title": "Senior Software Engineer",
      "company": "Google",
      "location": "San Francisco, CA",
      "score": 0.95,
      "salary": 150000,
      "job_type": "fulltime",
      "is_remote": false,
      "url": "https://jobs.google.com/job/456",
      "description": "We are looking for a Senior Software Engineer..."
    }
  ],
  "total_recommendations": 10,
  "model_version": "1.0.0",
  "timestamp": "2023-12-07T10:30:00Z",
  "processing_time_ms": 45.2
}
```

#### Error Responses

**Model Not Loaded (503)**
```json
{
  "detail": "Model not loaded"
}
```

**Invalid Request (422)**
```json
{
  "detail": [
    {
      "loc": ["body", "user_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Internal Server Error (500)**
```json
{
  "detail": "Error generating recommendations: [error message]"
}
```

### 5. Batch Recommendations

**POST** `/batch_recommend`

Get recommendations for multiple users in a single request.

#### Request Body
```json
{
  "user_ids": [123, 456, 789],
  "max_recommendations": 5,
  "filters": {
    "location_preference": "San Francisco",
    "salary_min": 100000
  }
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_ids` | array[integer] | Yes | - | List of user identifiers |
| `max_recommendations` | integer | No | 5 | Maximum recommendations per user (1-20) |
| `filters` | object | No | - | Common filters for all users |

#### Response
```json
{
  "batch_results": [
    {
      "user_id": 123,
      "recommendations": [...],
      "total_recommendations": 5
    },
    {
      "user_id": 456,
      "recommendations": [...],
      "total_recommendations": 5
    }
  ],
  "total_users": 3,
  "model_version": "1.0.0",
  "timestamp": "2023-12-07T10:30:00Z",
  "processing_time_ms": 125.8
}
```

### 6. User-Specific Recommendations

**GET** `/recommendations/{user_id}`

Get recommendations for a specific user (simplified endpoint).

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | integer | Yes | User identifier |

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 10 | Number of recommendations |

#### Response
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "job_id": 456,
      "title": "Software Engineer",
      "company": "Google",
      "location": "San Francisco, CA",
      "score": 0.92
    }
  ],
  "count": 10
}
```

### 7. Model Training

**POST** `/train`

Start training a new recommendation model (runs in background).

#### Request Body
```json
{
  "training_data_path": "data",
  "validation_split": 0.2,
  "epochs": 50,
  "batch_size": 32
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `training_data_path` | string | Yes | - | Path to training data directory |
| `validation_split` | number | No | 0.2 | Validation data split (0.1-0.5) |
| `epochs` | integer | No | 50 | Number of training epochs (1-200) |
| `batch_size` | integer | No | 32 | Training batch size (8-128) |

#### Response
```json
{
  "message": "Model training started",
  "training_data_path": "data",
  "epochs": 50,
  "batch_size": 32,
  "timestamp": "2023-12-07T10:30:00Z"
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request successful |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error - Server error |
| 503 | Service Unavailable - Model not loaded |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Scenarios

1. **Model Not Loaded**
   - Status: 503
   - Cause: API started without loading a trained model
   - Solution: Load a trained model or start training

2. **Validation Errors**
   - Status: 422
   - Cause: Invalid request parameters
   - Solution: Check request format and parameter values

3. **Internal Errors**
   - Status: 500
   - Cause: Unexpected server error
   - Solution: Check server logs for details

## Rate Limiting

Currently, no rate limiting is implemented. For production deployments, implement appropriate rate limiting based on your requirements.

## CORS Support

The API includes CORS middleware to allow cross-origin requests from web applications.

## Interactive Documentation

The API provides interactive documentation at:

- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Usage Examples

### Python Client

```python
import requests
import json

# API base URL
base_url = "http://localhost:8001"

# Get recommendations for a user
def get_recommendations(user_id, max_recommendations=10, **filters):
    url = f"{base_url}/recommend"
    data = {
        "user_id": user_id,
        "max_recommendations": max_recommendations,
        **filters
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Usage
recommendations = get_recommendations(
    user_id=123,
    max_recommendations=5,
    location_preference="San Francisco",
    salary_min=100000
)

print(f"Found {len(recommendations['recommendations'])} recommendations")
for rec in recommendations['recommendations']:
    print(f"- {rec['title']} at {rec['company']} (Score: {rec['score']:.3f})")
```

### JavaScript Client

```javascript
// Get recommendations for a user
async function getRecommendations(userId, maxRecommendations = 10, filters = {}) {
    const url = 'http://localhost:8001/recommend';
    const data = {
        user_id: userId,
        max_recommendations: maxRecommendations,
        ...filters
    };
    
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} - ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw error;
    }
}

// Usage
getRecommendations(123, 5, {
    location_preference: "San Francisco",
    salary_min: 100000
})
.then(recommendations => {
    console.log(`Found ${recommendations.recommendations.length} recommendations`);
    recommendations.recommendations.forEach(rec => {
        console.log(`- ${rec.title} at ${rec.company} (Score: ${rec.score.toFixed(3)})`);
    });
})
.catch(error => {
    console.error('Error:', error);
});
```

### cURL Examples

```bash
# Get recommendations
curl -X POST "http://localhost:8001/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": 123,
       "max_recommendations": 5,
       "location_preference": "San Francisco",
       "salary_min": 100000
     }'

# Check API health
curl -X GET "http://localhost:8001/health"

# Get model information
curl -X GET "http://localhost:8001/model_info"

# Get user-specific recommendations
curl -X GET "http://localhost:8001/recommendations/123?limit=5"
```

## Performance Considerations

### Response Times

- **Single User Recommendations**: <100ms
- **Batch Recommendations (10 users)**: <500ms
- **Health Check**: <10ms

### Optimization Tips

1. **Batch Requests**: Use batch endpoints for multiple users
2. **Caching**: Implement client-side caching for repeated requests
3. **Connection Pooling**: Reuse HTTP connections
4. **Async Requests**: Use asynchronous requests for better performance

## Monitoring and Logging

### Health Monitoring

Monitor the `/health` endpoint to ensure API availability:

```bash
# Simple health check
curl -f http://localhost:8001/health || echo "API is down"
```

### Logging

The API logs all requests and responses. Check the logs for:
- Request/response times
- Error messages
- Performance metrics
- User activity

### Metrics

Key metrics to monitor:
- Request rate (requests per second)
- Response times (p50, p95, p99)
- Error rates
- Model inference times
- Memory usage

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t job-recommendation-api .

# Run container
docker run -p 8001:8001 job-recommendation-api
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | 0.0.0.0 | API server host |
| `API_PORT` | 8001 | API server port |
| `MODEL_PATH` | models/best_convfm_model.pt | Path to trained model |
| `CONFIG_PATH` | artifacts/model_config.json | Path to model config |
| `LOG_LEVEL` | INFO | Logging level |

### Production Considerations

1. **Load Balancing**: Use multiple API instances behind a load balancer
2. **SSL/TLS**: Enable HTTPS for secure communication
3. **Authentication**: Implement proper authentication mechanisms
4. **Rate Limiting**: Add rate limiting to prevent abuse
5. **Monitoring**: Set up comprehensive monitoring and alerting
6. **Backup**: Regular backup of models and configurations
