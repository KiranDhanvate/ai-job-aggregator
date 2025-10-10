"""
Unit tests for feature extractor
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import the feature extractor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.feature_extractor import JobFeatureExtractor, JobDataset, create_dataloader


class TestJobFeatureExtractor:
    """Test cases for JobFeatureExtractor"""
    
    @pytest.fixture
    def sample_jobs_df(self):
        """Create sample job data"""
        return pd.DataFrame({
            'job_id': [1, 2, 3, 4, 5],
            'title': ['Software Engineer', 'Data Scientist', 'ML Engineer', 'DevOps Engineer', 'Frontend Developer'],
            'company': ['Google', 'Microsoft', 'Amazon', 'Netflix', 'Meta'],
            'location': ['San Francisco', 'Seattle', 'Austin', 'Los Angeles', 'New York'],
            'description': [
                'Python, JavaScript, React development',
                'Machine learning, Python, TensorFlow, data analysis',
                'Deep learning, PyTorch, computer vision',
                'AWS, Docker, Kubernetes, CI/CD',
                'React, TypeScript, Node.js, web development'
            ],
            'salary': [120000, 130000, 140000, 125000, 115000],
            'job_type': ['fulltime', 'fulltime', 'fulltime', 'contract', 'fulltime'],
            'is_remote': [False, True, False, True, False]
        })
    
    @pytest.fixture
    def sample_interactions_df(self):
        """Create sample interaction data"""
        return pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'job_id': [1, 2, 1, 3, 2, 4, 3, 5, 1, 4],
            'rating': [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.8],
            'timestamp': ['2023-01-01'] * 10
        })
    
    @pytest.fixture
    def feature_extractor(self):
        """Create feature extractor instance"""
        return JobFeatureExtractor(max_skills=20, max_text_features=100)
    
    def test_feature_extractor_initialization(self, feature_extractor):
        """Test feature extractor initialization"""
        assert feature_extractor.max_skills == 20
        assert feature_extractor.max_text_features == 100
        assert not feature_extractor.is_fitted
    
    def test_fit_feature_extractor(self, feature_extractor, sample_jobs_df, sample_interactions_df):
        """Test fitting the feature extractor"""
        fitted_extractor = feature_extractor.fit(sample_jobs_df, sample_interactions_df)
        
        assert fitted_extractor.is_fitted
        assert fitted_extractor.user_encoder is not None
        assert fitted_extractor.job_encoder is not None
        assert fitted_extractor.skill_encoder is not None
        assert fitted_extractor.location_encoder is not None
        assert fitted_extractor.company_encoder is not None
    
    def test_transform_jobs(self, feature_extractor, sample_jobs_df, sample_interactions_df):
        """Test job feature transformation"""
        # Fit the extractor first
        feature_extractor.fit(sample_jobs_df, sample_interactions_df)
        
        # Transform jobs
        job_features = feature_extractor.transform_jobs(sample_jobs_df)
        
        # Check required features are present
        required_features = ['job_ids', 'locations', 'companies', 'salaries', 'skill_features', 'text_features']
        for feature in required_features:
            assert feature in job_features
        
        # Check feature shapes
        assert len(job_features['job_ids']) == len(sample_jobs_df)
        assert len(job_features['locations']) == len(sample_jobs_df)
        assert len(job_features['companies']) == len(sample_jobs_df)
        assert job_features['skill_features'].shape[0] == len(sample_jobs_df)
        assert job_features['skill_features'].shape[1] == feature_extractor.max_skills
    
    def test_transform_interactions(self, feature_extractor, sample_jobs_df, sample_interactions_df):
        """Test interaction feature transformation"""
        # Fit the extractor first
        feature_extractor.fit(sample_jobs_df, sample_interactions_df)
        
        # Transform interactions
        interaction_features = feature_extractor.transform_interactions(sample_interactions_df)
        
        # Check required features are present
        required_features = ['user_ids', 'job_ids', 'labels']
        for feature in required_features:
            assert feature in interaction_features
        
        # Check feature shapes
        assert len(interaction_features['user_ids']) == len(sample_interactions_df)
        assert len(interaction_features['job_ids']) == len(sample_interactions_df)
        assert len(interaction_features['labels']) == len(sample_interactions_df)
    
    def test_skill_extraction(self, feature_extractor):
        """Test skill extraction from text"""
        text = "We need a Python developer with React, JavaScript, and machine learning experience. AWS and Docker knowledge preferred."
        skills = feature_extractor._parse_skills_from_text(text)
        
        expected_skills = ['python', 'react', 'javascript', 'machine learning', 'aws', 'docker']
        
        for skill in expected_skills:
            assert skill in skills
    
    def test_skill_extraction_empty_text(self, feature_extractor):
        """Test skill extraction with empty text"""
        skills = feature_extractor._parse_skills_from_text("")
        assert skills == []
        
        skills = feature_extractor._parse_skills_from_text(None)
        assert skills == []
    
    def test_job_type_encoding(self, feature_extractor, sample_jobs_df):
        """Test job type encoding"""
        job_types = feature_extractor._encode_job_types(sample_jobs_df)
        
        # Check that job types are encoded as integers
        assert all(isinstance(jt, np.integer) for jt in job_types)
        assert len(job_types) == len(sample_jobs_df)
    
    def test_experience_level_encoding(self, feature_extractor, sample_jobs_df):
        """Test experience level encoding"""
        exp_levels = feature_extractor._encode_experience_levels(sample_jobs_df)
        
        # Check that experience levels are encoded as integers
        assert all(isinstance(el, np.integer) for el in exp_levels)
        assert len(exp_levels) == len(sample_jobs_df)
    
    def test_feature_extractor_save_load(self, feature_extractor, sample_jobs_df, sample_interactions_df):
        """Test saving and loading feature extractor"""
        # Fit the extractor
        feature_extractor.fit(sample_jobs_df, sample_interactions_df)
        
        # Save extractor
        feature_extractor.save('test_extractor.pkl')
        
        # Load extractor
        loaded_extractor = JobFeatureExtractor.load('test_extractor.pkl')
        
        # Check that loaded extractor is fitted
        assert loaded_extractor.is_fitted
        
        # Check that encoders are loaded
        assert loaded_extractor.user_encoder is not None
        assert loaded_extractor.job_encoder is not None
        
        # Clean up
        os.remove('test_extractor.pkl')
    
    def test_unfitted_extractor_error(self, feature_extractor, sample_jobs_df):
        """Test error when trying to transform without fitting"""
        with pytest.raises(ValueError, match="Feature extractor must be fitted before transform"):
            feature_extractor.transform_jobs(sample_jobs_df)


class TestJobDataset:
    """Test cases for JobDataset"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature dictionary"""
        return {
            'user_ids': np.array([1, 2, 3, 4]),
            'job_ids': np.array([10, 20, 30, 40]),
            'skill_ids': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            'location_ids': np.array([1, 2, 1, 2]),
            'company_ids': np.array([5, 10, 15, 20])
        }
    
    @pytest.fixture
    def sample_labels(self):
        """Create sample labels"""
        return np.array([1, 0, 1, 0])
    
    def test_dataset_initialization(self, sample_features):
        """Test dataset initialization"""
        dataset = JobDataset(sample_features)
        
        assert len(dataset) == 4
        assert dataset.features == sample_features
        assert dataset.labels is None
    
    def test_dataset_with_labels(self, sample_features, sample_labels):
        """Test dataset initialization with labels"""
        dataset = JobDataset(sample_features, sample_labels)
        
        assert len(dataset) == 4
        assert dataset.features == sample_features
        assert np.array_equal(dataset.labels, sample_labels)
    
    def test_dataset_getitem(self, sample_features, sample_labels):
        """Test dataset item retrieval"""
        dataset = JobDataset(sample_features, sample_labels)
        
        item = dataset[0]
        
        # Check that all features are present
        for key in sample_features.keys():
            assert key in item
        
        # Check that labels are present
        assert 'labels' in item
        
        # Check tensor types
        for key, value in item.items():
            if key == 'labels':
                assert value.dtype == torch.float32
            elif sample_features[key].dtype in [np.int32, np.int64]:
                assert value.dtype == torch.long
            else:
                assert value.dtype == torch.float32
    
    def test_dataset_length_mismatch(self):
        """Test error when feature arrays have different lengths"""
        features = {
            'user_ids': np.array([1, 2, 3]),
            'job_ids': np.array([10, 20])  # Different length
        }
        
        with pytest.raises(ValueError):
            JobDataset(features)


class TestDataLoader:
    """Test cases for DataLoader creation"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature dictionary"""
        return {
            'user_ids': np.array([1, 2, 3, 4, 5]),
            'job_ids': np.array([10, 20, 30, 40, 50]),
            'skill_ids': np.random.randint(0, 100, (5, 10)),
            'location_ids': np.array([1, 2, 1, 2, 1]),
            'company_ids': np.array([5, 10, 15, 20, 25])
        }
    
    def test_dataloader_creation(self, sample_features):
        """Test DataLoader creation"""
        dataloader = create_dataloader(sample_features, batch_size=2, shuffle=True)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 2
        
        # Test that we can iterate through the dataloader
        for batch in dataloader:
            assert len(batch['user_ids']) <= 2
            break
    
    def test_dataloader_with_labels(self, sample_features):
        """Test DataLoader creation with labels"""
        labels = np.array([1, 0, 1, 0, 1])
        dataloader = create_dataloader(sample_features, labels, batch_size=3)
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        
        # Test that labels are included
        for batch in dataloader:
            assert 'labels' in batch
            break


if __name__ == "__main__":
    pytest.main([__file__])
