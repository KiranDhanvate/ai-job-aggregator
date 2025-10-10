"""
ML Models Package for AI Job Aggregator

This package contains machine learning models and utilities for job recommendations.
"""

from .convfm_model import ConvFMJobRecommender
from .feature_extractor import JobFeatureExtractor
from .training_pipeline import ConvFMTrainingPipeline

__version__ = "1.0.0"
__all__ = ["ConvFMJobRecommender", "JobFeatureExtractor", "ConvFMTrainingPipeline"]
