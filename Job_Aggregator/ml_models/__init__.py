"""
Machine Learning Models for Job Recommendation System

This package contains the ConvFM (Hybrid CNN + Factorization Machine) 
implementation for personalized job recommendations.

Components:
- convfm_model: Neural network architecture
- feature_extractor: Feature engineering pipeline
- training_pipeline: Training and evaluation utilities
"""

__version__ = "1.0.0"

from ml_models.convfm_model import (
    ConvFM,
    TextCNN,
    FactorizationMachine,
    create_convfm_model,
    ConvFMLoss
)

from ml_models.feature_extractor import (
    FeatureExtractor,
    TextPreprocessor,
    DatasetBuilder
)

from ml_models.training_pipeline import (
    ConvFMTrainer,
    JobRecommendationDataset,
    prepare_training_data,
    create_sample_user_profiles
)

__all__ = [
    # Model components
    'ConvFM',
    'TextCNN',
    'FactorizationMachine',
    'create_convfm_model',
    'ConvFMLoss',
    
    # Feature engineering
    'FeatureExtractor',
    'TextPreprocessor',
    'DatasetBuilder',
    
    # Training utilities
    'ConvFMTrainer',
    'JobRecommendationDataset',
    'prepare_training_data',
    'create_sample_user_profiles'
]
