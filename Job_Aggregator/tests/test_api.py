"""
Unit tests for recommendation API
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import the API
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.recommendation_api import RecommendationAPI, create_app


class TestRecommendationAPI:
    """Test cases for RecommendationAPI"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        model = Mock()
        model.predict_job_scores.return_value = {
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.5
        }
        return model
    
    @pytest.fixture
    def mock_feature_extractor(self):
        """Create mock feature extractor"""
        extractor = Mock()
        extractor.user_encoder.classes_ = list(range(100))
        extractor.job_encoder.classes_ = list(range(1000))
        extractor.skill_encoder.classes_ = list(range(500))
        extractor.location_encoder.classes_ = list(range(50))
        extractor.company_encoder.classes_ = list(range(200))
        return extractor
    
    @pytest.fixture
    def mock_jobs_df(self):
        """Create mock jobs DataFrame"""
        import pandas as pd
        return pd.DataFrame({
            'job_id': [1, 2, 3, 4, 5],
            'title': ['Software Engineer', 'Data Scientist', 'ML Engineer', 'DevOps Engineer', 'Frontend Developer'],
            'company': ['Google', 'Microsoft', 'Amazon', 'Netflix', 'Meta'],
            'location': ['San Francisco', 'Seattle', 'Austin', 'Los Angeles', 'New York'],
            'salary': [120000, 130000, 140000, 125000, 115000],
            'job_type': ['fulltime', 'fulltime', 'fulltime', 'contract', 'fulltime'],
            'is_remote': [False, True, False, True, False],
            'job_url': ['url1', 'url2', 'url3', 'url4', 'url5'],
            'description': ['desc1', 'desc2', 'desc3', 'desc4', 'desc5']
        })
    
    @pytest.fixture
    def api_with_mocks(self, mock_model, mock_feature_extractor, mock_jobs_df):
        """Create API instance with mocked dependencies"""
        api = RecommendationAPI()
        api.model = mock_model
        api.feature_extractor = mock_feature_extractor
        api.jobs_df = mock_jobs_df
        api.model_loaded = True
        return api
    
    def test_api_initialization(self):
        """Test API initialization"""
        api = RecommendationAPI()
        
        assert api.model is None
        assert api.feature_extractor is None
        assert api.jobs_df is None
        assert not api.model_loaded
    
    def test_root_endpoint(self, api_with_mocks):
        """Test root endpoint"""
        client = TestClient(api_with_mocks.app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, api_with_mocks):
        """Test health check endpoint"""
        client = TestClient(api_with_mocks.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        assert "timestamp" in data
    
    def test_health_endpoint_model_not_loaded(self):
        """Test health endpoint when model is not loaded"""
        api = RecommendationAPI()
        client = TestClient(api.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == False
    
    def test_model_info_endpoint(self, api_with_mocks):
        """Test model info endpoint"""
        client = TestClient(api_with_mocks.app)
        response = client.get("/model_info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "ConvFM"
        assert "version" in data
        assert "num_users" in data
        assert "num_jobs" in data
        assert "num_skills" in data
    
    def test_model_info_endpoint_not_loaded(self):
        """Test model info endpoint when model is not loaded"""
        api = RecommendationAPI()
        client = TestClient(api.app)
        response = client.get("/model_info")
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_recommend_endpoint(self, api_with_mocks):
        """Test recommendation endpoint"""
        client = TestClient(api_with_mocks.app)
        
        request_data = {
            "user_id": 1,
            "max_recommendations": 5
        }
        
        response = client.post("/recommend", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == 1
        assert "recommendations" in data
        assert "total_recommendations" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data
        
        # Check recommendations format
        recommendations = data["recommendations"]
        assert len(recommendations) <= 5
        
        if recommendations:
            rec = recommendations[0]
            assert "job_id" in rec
            assert "title" in rec
            assert "company" in rec
            assert "location" in rec
            assert "score" in rec
    
    def test_recommend_endpoint_with_filters(self, api_with_mocks):
        """Test recommendation endpoint with filters"""
        client = TestClient(api_with_mocks.app)
        
        request_data = {
            "user_id": 1,
            "max_recommendations": 3,
            "location_preference": "San Francisco",
            "salary_min": 100000,
            "is_remote": False
        }
        
        response = client.post("/recommend", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
    
    def test_recommend_endpoint_model_not_loaded(self):
        """Test recommendation endpoint when model is not loaded"""
        api = RecommendationAPI()
        client = TestClient(api.app)
        
        request_data = {
            "user_id": 1,
            "max_recommendations": 5
        }
        
        response = client.post("/recommend", json=request_data)
        
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]
    
    def test_batch_recommend_endpoint(self, api_with_mocks):
        """Test batch recommendation endpoint"""
        client = TestClient(api_with_mocks.app)
        
        request_data = {
            "user_ids": [1, 2, 3],
            "max_recommendations": 3
        }
        
        response = client.post("/batch_recommend", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "batch_results" in data
        assert "total_users" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data
        
        batch_results = data["batch_results"]
        assert len(batch_results) == 3
        
        for result in batch_results:
            assert "user_id" in result
            assert "recommendations" in result
            assert "total_recommendations" in result
    
    def test_user_recommendations_endpoint(self, api_with_mocks):
        """Test user-specific recommendations endpoint"""
        client = TestClient(api_with_mocks.app)
        response = client.get("/recommendations/1")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == 1
        assert "recommendations" in data
        assert "count" in data
    
    def test_train_endpoint(self, api_with_mocks):
        """Test training endpoint"""
        client = TestClient(api_with_mocks.app)
        
        request_data = {
            "training_data_path": "data",
            "validation_split": 0.2,
            "epochs": 10,
            "batch_size": 16
        }
        
        response = client.post("/train", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "training_data_path" in data
        assert "epochs" in data
        assert "batch_size" in data
        assert "timestamp" in data
    
    def test_cors_headers(self, api_with_mocks):
        """Test CORS headers"""
        client = TestClient(api_with_mocks.app)
        response = client.options("/")
        
        # CORS preflight request should be handled
        assert response.status_code in [200, 204]
    
    def test_invalid_request_data(self, api_with_mocks):
        """Test handling of invalid request data"""
        client = TestClient(api_with_mocks.app)
        
        # Invalid user_id (should be integer)
        request_data = {
            "user_id": "invalid",
            "max_recommendations": 5
        }
        
        response = client.post("/recommend", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_missing_required_fields(self, api_with_mocks):
        """Test handling of missing required fields"""
        client = TestClient(api_with_mocks.app)
        
        # Missing user_id
        request_data = {
            "max_recommendations": 5
        }
        
        response = client.post("/recommend", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_max_recommendations_limit(self, api_with_mocks):
        """Test max recommendations limit"""
        client = TestClient(api_with_mocks.app)
        
        # Request more than maximum allowed
        request_data = {
            "user_id": 1,
            "max_recommendations": 100  # Should be limited to 50
        }
        
        response = client.post("/recommend", json=request_data)
        
        # Should return validation error
        assert response.status_code == 422


class TestAPIUtilities:
    """Test utility functions"""
    
    def test_filter_jobs_functionality(self):
        """Test job filtering functionality"""
        api = RecommendationAPI()
        
        import pandas as pd
        jobs_df = pd.DataFrame({
            'job_id': [1, 2, 3, 4, 5],
            'title': ['Engineer A', 'Engineer B', 'Engineer C', 'Engineer D', 'Engineer E'],
            'company': ['Google', 'Microsoft', 'Amazon', 'Netflix', 'Meta'],
            'location': ['San Francisco', 'Seattle', 'San Francisco', 'Los Angeles', 'New York'],
            'salary': [120000, 130000, 140000, 125000, 115000],
            'job_type': ['fulltime', 'fulltime', 'contract', 'fulltime', 'fulltime'],
            'is_remote': [False, True, False, True, False]
        })
        
        api.jobs_df = jobs_df
        
        # Test location filtering
        filtered = api._filter_jobs(location_preference="San Francisco")
        assert len(filtered) == 2
        assert all(loc == 'San Francisco' for loc in filtered['location'])
        
        # Test salary filtering
        filtered = api._filter_jobs(salary_min=125000)
        assert len(filtered) == 3
        assert all(sal >= 125000 for sal in filtered['salary'])
        
        # Test remote filtering
        filtered = api._filter_jobs(is_remote=True)
        assert len(filtered) == 2
        assert all(remote == True for remote in filtered['is_remote'])
        
        # Test combined filtering
        filtered = api._filter_jobs(
            location_preference="San Francisco",
            salary_min=130000
        )
        assert len(filtered) == 1
        assert filtered.iloc[0]['location'] == 'San Francisco'
        assert filtered.iloc[0]['salary'] >= 130000
    
    def test_create_app_function(self):
        """Test create_app function"""
        app = create_app()
        
        assert app is not None
        assert hasattr(app, 'routes')
        
        # Test that the app has the expected routes
        route_paths = [route.path for route in app.routes]
        expected_paths = ["/", "/health", "/model_info", "/recommend", "/batch_recommend"]
        
        for path in expected_paths:
            assert path in route_paths


if __name__ == "__main__":
    pytest.main([__file__])
