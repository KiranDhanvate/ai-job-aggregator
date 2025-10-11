"""
Unit tests for ConvFM model
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.convfm_model import ConvFMJobRecommender, ConvFMTrainer, create_model_from_config


class TestConvFMJobRecommender:
    """Test cases for ConvFMJobRecommender model"""
    
    @pytest.fixture
    def model_config(self):
        """Sample model configuration"""
        return {
            'num_users': 100,
            'num_jobs': 1000,
            'num_skills': 500,
            'num_locations': 50,
            'num_companies': 200,
            'embedding_dim': 32,
            'conv_filters': 32,
            'conv_kernel_size': 3,
            'dropout_rate': 0.2,
            'hidden_dims': [64, 32]
        }
    
    @pytest.fixture
    def sample_model(self, model_config):
        """Create a sample model for testing"""
        return ConvFMJobRecommender(**model_config)
    
    def test_model_initialization(self, model_config):
        """Test model initialization"""
        model = ConvFMJobRecommender(**model_config)
        
        assert model.num_users == model_config['num_users']
        assert model.num_jobs == model_config['num_jobs']
        assert model.embedding_dim == model_config['embedding_dim']
        
        # Check embedding layers
        assert model.user_embedding.num_embeddings == model_config['num_users']
        assert model.job_embedding.num_embeddings == model_config['num_jobs']
        assert model.skill_embedding.num_embeddings == model_config['num_skills']
    
    def test_forward_pass(self, sample_model):
        """Test forward pass with sample data"""
        batch_size = 4
        max_skills = 10
        
        # Create sample input tensors
        user_ids = torch.randint(0, sample_model.num_users, (batch_size,))
        job_ids = torch.randint(0, sample_model.num_jobs, (batch_size,))
        skill_ids = torch.randint(0, sample_model.skill_encoder.num_embeddings, (batch_size, max_skills))
        location_ids = torch.randint(0, sample_model.location_encoder.num_embeddings, (batch_size,))
        company_ids = torch.randint(0, sample_model.company_encoder.num_embeddings, (batch_size,))
        
        # Forward pass
        output = sample_model(user_ids, job_ids, skill_ids, location_ids, company_ids)
        
        # Check output shape and range
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_predict_job_scores(self, sample_model):
        """Test job score prediction"""
        user_id = 1
        job_ids = [1, 2, 3, 4, 5]
        skill_ids = [10, 20, 30]
        location_id = 1
        company_id = 5
        
        scores = sample_model.predict_job_scores(
            user_id, job_ids, skill_ids, location_id, company_id
        )
        
        # Check output format
        assert len(scores) == len(job_ids)
        assert all(isinstance(score, float) for score in scores.values())
        assert all(0 <= score <= 1 for score in scores.values())
        
        # Check all job IDs are present
        assert set(scores.keys()) == set(job_ids)
    
    def test_embedding_retrieval(self, sample_model):
        """Test embedding retrieval methods"""
        user_ids = torch.tensor([1, 2, 3])
        job_ids = torch.tensor([10, 20, 30])
        
        user_embeddings = sample_model.get_user_embeddings(user_ids)
        job_embeddings = sample_model.get_job_embeddings(job_ids)
        
        assert user_embeddings.shape == (3, sample_model.embedding_dim)
        assert job_embeddings.shape == (3, sample_model.embedding_dim)
    
    def test_model_creation_from_config(self, model_config):
        """Test model creation from config"""
        model = create_model_from_config(model_config)
        
        assert isinstance(model, ConvFMJobRecommender)
        assert model.num_users == model_config['num_users']
        assert model.num_jobs == model_config['num_jobs']


class TestConvFMTrainer:
    """Test cases for ConvFMTrainer"""
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for training tests"""
        return ConvFMJobRecommender(
            num_users=50,
            num_jobs=500,
            num_skills=200,
            num_locations=20,
            num_companies=100,
            embedding_dim=16,
            conv_filters=16,
            hidden_dims=[32, 16]
        )
    
    @pytest.fixture
    def sample_trainer(self, sample_model):
        """Create sample trainer"""
        return ConvFMTrainer(sample_model, learning_rate=0.001)
    
    @pytest.fixture
    def sample_dataloader(self):
        """Create sample dataloader"""
        from torch.utils.data import DataLoader, TensorDataset
        
        batch_size = 4
        user_ids = torch.randint(0, 50, (batch_size,))
        job_ids = torch.randint(0, 500, (batch_size,))
        skill_ids = torch.randint(0, 200, (batch_size, 10))
        location_ids = torch.randint(0, 20, (batch_size,))
        company_ids = torch.randint(0, 100, (batch_size,))
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        dataset = TensorDataset(user_ids, job_ids, skill_ids, location_ids, company_ids, labels)
        return DataLoader(dataset, batch_size=2)
    
    def test_trainer_initialization(self, sample_trainer):
        """Test trainer initialization"""
        assert sample_trainer.model is not None
        assert sample_trainer.optimizer is not None
        assert sample_trainer.criterion is not None
        assert sample_trainer.device is not None
    
    def test_train_epoch(self, sample_trainer, sample_dataloader):
        """Test training for one epoch"""
        # Mock the dataloader to return proper format
        mock_batch = {
            'user_ids': torch.randint(0, 50, (2,)),
            'job_ids': torch.randint(0, 500, (2,)),
            'skill_ids': torch.randint(0, 200, (2, 10)),
            'location_ids': torch.randint(0, 20, (2,)),
            'company_ids': torch.randint(0, 100, (2,)),
            'labels': torch.randint(0, 2, (2,)).float()
        }
        
        mock_dataloader = [mock_batch]
        
        # Train for one epoch
        loss = sample_trainer.train_epoch(mock_dataloader)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_evaluate(self, sample_trainer, sample_dataloader):
        """Test model evaluation"""
        # Mock the dataloader
        mock_batch = {
            'user_ids': torch.randint(0, 50, (2,)),
            'job_ids': torch.randint(0, 500, (2,)),
            'skill_ids': torch.randint(0, 200, (2, 10)),
            'location_ids': torch.randint(0, 20, (2,)),
            'company_ids': torch.randint(0, 100, (2,)),
            'labels': torch.randint(0, 2, (2,)).float()
        }
        
        mock_dataloader = [mock_batch]
        
        # Evaluate
        metrics = sample_trainer.evaluate(mock_dataloader)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'predictions' in metrics
        assert 'labels' in metrics
        
        assert isinstance(metrics['loss'], float)
        assert isinstance(metrics['accuracy'], (float, np.floating))
        assert len(metrics['predictions']) == 2
        assert len(metrics['labels']) == 2


class TestModelIntegration:
    """Integration tests for the complete model"""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline"""
        model = ConvFMJobRecommender(
            num_users=100,
            num_jobs=1000,
            num_skills=500,
            num_locations=50,
            num_companies=200,
            embedding_dim=32
        )
        
        # Test prediction
        user_id = 1
        job_ids = [1, 2, 3]
        skill_ids = [10, 20]
        location_id = 1
        company_id = 5
        
        scores = model.predict_job_scores(
            user_id, job_ids, skill_ids, location_id, company_id
        )
        
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores.values())
    
    def test_model_serialization(self):
        """Test model saving and loading"""
        model = ConvFMJobRecommender(
            num_users=50,
            num_jobs=500,
            num_skills=200,
            num_locations=20,
            num_companies=100,
            embedding_dim=16
        )
        
        # Save model
        torch.save(model.state_dict(), 'test_model.pt')
        
        # Load model
        new_model = ConvFMJobRecommender(
            num_users=50,
            num_jobs=500,
            num_skills=200,
            num_locations=20,
            num_companies=100,
            embedding_dim=16
        )
        
        new_model.load_state_dict(torch.load('test_model.pt'))
        
        # Test that both models produce same output
        user_ids = torch.tensor([1, 2])
        job_ids = torch.tensor([10, 20])
        skill_ids = torch.randint(0, 200, (2, 10))
        location_ids = torch.tensor([1, 2])
        company_ids = torch.tensor([5, 10])
        
        with torch.no_grad():
            output1 = model(user_ids, job_ids, skill_ids, location_ids, company_ids)
            output2 = new_model(user_ids, job_ids, skill_ids, location_ids, company_ids)
            
            assert torch.allclose(output1, output2, atol=1e-6)
        
        # Clean up
        os.remove('test_model.pt')


if __name__ == "__main__":
    pytest.main([__file__])
