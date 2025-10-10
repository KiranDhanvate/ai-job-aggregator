"""
Recommendation API for AI Job Aggregator

This module provides FastAPI endpoints for job recommendations using the ConvFM model.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import ML models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.convfm_model import ConvFMJobRecommender
from ml_models.feature_extractor import JobFeatureExtractor
from ml_models.training_pipeline import ConvFMTrainingPipeline

logger = logging.getLogger(__name__)

# Pydantic models for API
class JobRecommendationRequest(BaseModel):
    """Request model for job recommendations"""
    user_id: int = Field(..., description="User ID")
    max_recommendations: int = Field(default=10, ge=1, le=50, description="Maximum number of recommendations")
    location_preference: Optional[str] = Field(None, description="Preferred location")
    company_preference: Optional[str] = Field(None, description="Preferred company")
    salary_min: Optional[float] = Field(None, ge=0, description="Minimum salary")
    job_type: Optional[str] = Field(None, description="Job type preference")
    is_remote: Optional[bool] = Field(None, description="Remote work preference")

class JobRecommendationResponse(BaseModel):
    """Response model for job recommendations"""
    user_id: int
    recommendations: List[Dict[str, Any]]
    total_recommendations: int
    model_version: str
    timestamp: str
    processing_time_ms: float

class BatchRecommendationRequest(BaseModel):
    """Request model for batch job recommendations"""
    user_ids: List[int] = Field(..., description="List of user IDs")
    max_recommendations: int = Field(default=5, ge=1, le=20, description="Maximum recommendations per user")
    filters: Optional[Dict[str, Any]] = Field(None, description="Common filters for all users")

class ModelTrainingRequest(BaseModel):
    """Request model for model training"""
    training_data_path: str = Field(..., description="Path to training data")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation data split")
    epochs: int = Field(default=50, ge=1, le=200, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=128, description="Training batch size")

class RecommendationAPI:
    """
    FastAPI application for job recommendations
    """
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.app = FastAPI(
            title="AI Job Aggregator Recommendation API",
            description="API for AI-powered job recommendations using ConvFM model",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize model and components
        self.model = None
        self.feature_extractor = None
        self.jobs_df = None
        self.model_version = "1.0.0"
        self.model_loaded = False
        
        # Setup routes
        self._setup_routes()
        
        # Load model if paths provided
        if model_path and config_path:
            self.load_model(model_path, config_path)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information"""
            return {
                "message": "AI Job Aggregator Recommendation API",
                "version": self.model_version,
                "status": "healthy" if self.model_loaded else "model_not_loaded",
                "endpoints": {
                    "/recommend": "POST - Get job recommendations for a user",
                    "/batch_recommend": "POST - Get batch recommendations",
                    "/health": "GET - Health check",
                    "/model_info": "GET - Model information",
                    "/train": "POST - Train new model",
                    "/docs": "GET - API documentation"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": self.model_loaded,
                "model_version": self.model_version
            }
        
        @self.app.get("/model_info")
        async def model_info():
            """Get model information"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            return {
                "model_type": "ConvFM",
                "version": self.model_version,
                "num_users": len(self.feature_extractor.user_encoder.classes_) if self.feature_extractor else 0,
                "num_jobs": len(self.feature_extractor.job_encoder.classes_) if self.feature_extractor else 0,
                "num_skills": len(self.feature_extractor.skill_encoder.classes_) if self.feature_extractor else 0,
                "loaded_at": datetime.now().isoformat()
            }
        
        @self.app.post("/recommend", response_model=JobRecommendationResponse)
        async def get_recommendations(request: JobRecommendationRequest):
            """Get job recommendations for a user"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = datetime.now()
            
            try:
                recommendations = await self._generate_recommendations(
                    user_id=request.user_id,
                    max_recommendations=request.max_recommendations,
                    location_preference=request.location_preference,
                    company_preference=request.company_preference,
                    salary_min=request.salary_min,
                    job_type=request.job_type,
                    is_remote=request.is_remote
                )
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return JobRecommendationResponse(
                    user_id=request.user_id,
                    recommendations=recommendations,
                    total_recommendations=len(recommendations),
                    model_version=self.model_version,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=processing_time
                )
                
            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
        
        @self.app.post("/batch_recommend")
        async def batch_recommendations(request: BatchRecommendationRequest):
            """Get batch job recommendations for multiple users"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = datetime.now()
            
            try:
                batch_results = []
                
                for user_id in request.user_ids:
                    recommendations = await self._generate_recommendations(
                        user_id=user_id,
                        max_recommendations=request.max_recommendations,
                        **request.filters or {}
                    )
                    
                    batch_results.append({
                        "user_id": user_id,
                        "recommendations": recommendations,
                        "total_recommendations": len(recommendations)
                    })
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    "batch_results": batch_results,
                    "total_users": len(request.user_ids),
                    "model_version": self.model_version,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"Error in batch recommendations: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in batch recommendations: {str(e)}")
        
        @self.app.post("/train")
        async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
            """Train a new recommendation model"""
            try:
                # Add training task to background
                background_tasks.add_task(
                    self._train_model_background,
                    request.training_data_path,
                    request.validation_split,
                    request.epochs,
                    request.batch_size
                )
                
                return {
                    "message": "Model training started",
                    "training_data_path": request.training_data_path,
                    "epochs": request.epochs,
                    "batch_size": request.batch_size,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error starting model training: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")
        
        @self.app.get("/recommendations/{user_id}")
        async def get_user_recommendations(user_id: int, limit: int = 10):
            """Get recommendations for a specific user (simplified endpoint)"""
            if not self.model_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            try:
                recommendations = await self._generate_recommendations(
                    user_id=user_id,
                    max_recommendations=limit
                )
                
                return {
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "count": len(recommendations)
                }
                
            except Exception as e:
                logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_recommendations(
        self,
        user_id: int,
        max_recommendations: int = 10,
        location_preference: Optional[str] = None,
        company_preference: Optional[str] = None,
        salary_min: Optional[float] = None,
        job_type: Optional[str] = None,
        is_remote: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Generate job recommendations for a user"""
        
        # Filter jobs based on preferences
        filtered_jobs = self._filter_jobs(
            location_preference=location_preference,
            company_preference=company_preference,
            salary_min=salary_min,
            job_type=job_type,
            is_remote=is_remote
        )
        
        if len(filtered_jobs) == 0:
            return []
        
        # Get job IDs for prediction
        job_ids = filtered_jobs['job_id'].tolist()
        
        # Generate user profile (simplified - would need actual user data)
        user_skills = self._get_user_skills(user_id)
        location_id = self._get_location_id(location_preference)
        company_id = self._get_company_id(company_preference)
        
        # Get model predictions
        scores = self.model.predict_job_scores(
            user_id=user_id,
            job_ids=job_ids,
            skill_ids=user_skills,
            location_id=location_id,
            company_id=company_id
        )
        
        # Sort by score and get top recommendations
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_job_ids = [job_id for job_id, score in sorted_scores[:max_recommendations]]
        
        # Format recommendations
        recommendations = []
        for job_id, score in sorted_scores[:max_recommendations]:
            job_data = filtered_jobs[filtered_jobs['job_id'] == job_id].iloc[0]
            
            recommendation = {
                "job_id": int(job_id),
                "title": job_data['title'],
                "company": job_data['company'],
                "location": job_data['location'],
                "score": float(score),
                "salary": float(job_data.get('salary', 0)) if pd.notna(job_data.get('salary')) else None,
                "job_type": job_data.get('job_type'),
                "is_remote": bool(job_data.get('is_remote', False)),
                "url": job_data.get('job_url'),
                "description": job_data.get('description', '')[:200] + "..." if len(str(job_data.get('description', ''))) > 200 else job_data.get('description', '')
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _filter_jobs(
        self,
        location_preference: Optional[str] = None,
        company_preference: Optional[str] = None,
        salary_min: Optional[float] = None,
        job_type: Optional[str] = None,
        is_remote: Optional[bool] = None
    ) -> pd.DataFrame:
        """Filter jobs based on preferences"""
        
        if self.jobs_df is None:
            return pd.DataFrame()
        
        filtered_jobs = self.jobs_df.copy()
        
        # Apply filters
        if location_preference:
            filtered_jobs = filtered_jobs[
                filtered_jobs['location'].str.contains(location_preference, case=False, na=False)
            ]
        
        if company_preference:
            filtered_jobs = filtered_jobs[
                filtered_jobs['company'].str.contains(company_preference, case=False, na=False)
            ]
        
        if salary_min is not None:
            filtered_jobs = filtered_jobs[filtered_jobs['salary'] >= salary_min]
        
        if job_type:
            filtered_jobs = filtered_jobs[filtered_jobs['job_type'] == job_type]
        
        if is_remote is not None:
            filtered_jobs = filtered_jobs[filtered_jobs['is_remote'] == is_remote]
        
        return filtered_jobs
    
    def _get_user_skills(self, user_id: int) -> List[int]:
        """Get user skills (simplified implementation)"""
        # This would typically query a user profile database
        # For now, return empty list
        return []
    
    def _get_location_id(self, location: Optional[str]) -> int:
        """Get location ID from location name"""
        if location and self.feature_extractor:
            try:
                return self.feature_extractor.location_encoder.transform([location])[0]
            except ValueError:
                return 0
        return 0
    
    def _get_company_id(self, company: Optional[str]) -> int:
        """Get company ID from company name"""
        if company and self.feature_extractor:
            try:
                return self.feature_extractor.company_encoder.transform([company])[0]
            except ValueError:
                return 0
        return 0
    
    def load_model(self, model_path: str, config_path: str):
        """Load trained model and feature extractor"""
        try:
            # Load model configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load feature extractor
            extractor_path = config_path.replace('_config.json', '_extractor.pkl')
            self.feature_extractor = JobFeatureExtractor.load(extractor_path)
            
            # Initialize model
            model_config = {
                'num_users': len(self.feature_extractor.user_encoder.classes_),
                'num_jobs': len(self.feature_extractor.job_encoder.classes_),
                'num_skills': len(self.feature_extractor.skill_encoder.classes_),
                'num_locations': len(self.feature_extractor.location_encoder.classes_),
                'num_companies': len(self.feature_extractor.company_encoder.classes_),
                'embedding_dim': config.get('embedding_dim', 64),
                'conv_filters': config.get('conv_filters', 64),
                'conv_kernel_size': config.get('conv_kernel_size', 3),
                'dropout_rate': config.get('dropout_rate', 0.2),
                'hidden_dims': config.get('hidden_dims', [128, 64, 32])
            }
            
            self.model = ConvFMJobRecommender(**model_config)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            
            # Load job data
            self._load_job_data()
            
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_job_data(self):
        """Load job data for recommendations"""
        try:
            # Try to load from data directory
            data_path = "data/collected_jobs.csv"
            if os.path.exists(data_path):
                self.jobs_df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(self.jobs_df)} jobs from {data_path}")
            else:
                logger.warning("Job data file not found. Recommendations may be limited.")
                self.jobs_df = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading job data: {str(e)}")
            self.jobs_df = pd.DataFrame()
    
    async def _train_model_background(
        self,
        training_data_path: str,
        validation_split: float,
        epochs: int,
        batch_size: int
    ):
        """Background task for model training"""
        try:
            logger.info("Starting background model training...")
            
            # Load training data
            jobs_df = pd.read_csv(os.path.join(training_data_path, 'jobs.csv'))
            interactions_df = pd.read_csv(os.path.join(training_data_path, 'interactions.csv'))
            
            # Create training configuration
            config = {
                'embedding_dim': 64,
                'batch_size': batch_size,
                'num_epochs': epochs,
                'test_size': validation_split,
                'learning_rate': 0.001,
                'models_dir': 'models',
                'artifacts_dir': 'artifacts',
                'logs_dir': 'logs'
            }
            
            # Initialize training pipeline
            pipeline = ConvFMTrainingPipeline(config)
            
            # Run training
            results = pipeline.run_full_pipeline(jobs_df, interactions_df)
            
            logger.info("Background model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in background training: {str(e)}")


def create_app(model_path: Optional[str] = None, config_path: Optional[str] = None) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        model_path: Path to trained model file
        config_path: Path to model configuration file
        
    Returns:
        Configured FastAPI application
    """
    api = RecommendationAPI(model_path, config_path)
    return api.app


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
