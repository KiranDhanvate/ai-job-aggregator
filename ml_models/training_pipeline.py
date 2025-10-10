"""
Training Pipeline for ConvFM Job Recommender

This module provides a complete training pipeline for the ConvFM job recommendation model.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle

from .convfm_model import ConvFMJobRecommender, ConvFMTrainer, create_model_from_config
from .feature_extractor import JobFeatureExtractor, create_dataloader, prepare_training_data

logger = logging.getLogger(__name__)


class ConvFMTrainingPipeline:
    """
    Complete training pipeline for ConvFM job recommender
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.feature_extractor = None
        self.model = None
        self.trainer = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epochs': []
        }
        
        # Create output directories
        self.models_dir = config.get('models_dir', 'models')
        self.artifacts_dir = config.get('artifacts_dir', 'artifacts')
        self.logs_dir = config.get('logs_dir', 'logs')
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def prepare_data(self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Tuple:
        """
        Prepare training and validation data
        
        Args:
            jobs_df: Job data DataFrame
            interactions_df: User-job interactions DataFrame
            
        Returns:
            Tuple of (train_loader, val_loader, feature_extractor)
        """
        logger.info("Preparing training data...")
        
        # Prepare data
        train_features, val_features, train_labels, val_labels, feature_extractor = prepare_training_data(
            jobs_df, interactions_df, 
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_features, train_labels,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = create_dataloader(
            val_features, val_labels,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )
        
        self.feature_extractor = feature_extractor
        
        logger.info(f"Data prepared: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")
        
        return train_loader, val_loader, feature_extractor
    
    def initialize_model(self) -> ConvFMJobRecommender:
        """
        Initialize the ConvFM model
        
        Returns:
            Initialized ConvFM model
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor must be fitted before initializing model")
        
        # Get feature dimensions from extractor
        model_config = {
            'num_users': len(self.feature_extractor.user_encoder.classes_),
            'num_jobs': len(self.feature_extractor.job_encoder.classes_),
            'num_skills': len(self.feature_extractor.skill_encoder.classes_),
            'num_locations': len(self.feature_extractor.location_encoder.classes_),
            'num_companies': len(self.feature_extractor.company_encoder.classes_),
            'embedding_dim': self.config.get('embedding_dim', 64),
            'conv_filters': self.config.get('conv_filters', 64),
            'conv_kernel_size': self.config.get('conv_kernel_size', 3),
            'dropout_rate': self.config.get('dropout_rate', 0.2),
            'hidden_dims': self.config.get('hidden_dims', [128, 64, 32])
        }
        
        # Create model
        self.model = create_model_from_config(model_config)
        self.model.to(self.device)
        
        # Initialize trainer
        self.trainer = ConvFMTrainer(
            self.model,
            learning_rate=self.config.get('learning_rate', 0.001)
        )
        
        logger.info("Model initialized successfully")
        return self.model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the ConvFM model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history dictionary
        """
        if self.model is None or self.trainer is None:
            raise ValueError("Model must be initialized before training")
        
        num_epochs = num_epochs or self.config.get('num_epochs', 50)
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.trainer.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.trainer.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            
            # Store history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(0.0)  # Could be calculated
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_best_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training completed successfully")
        return self.training_history
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        metrics = self.trainer.evaluate(test_loader)
        
        # Calculate additional metrics
        predictions = metrics['predictions']
        labels = metrics['labels']
        
        # Convert continuous labels to binary (ratings > 0.5 are positive)
        labels_binary = (labels > 0.5).astype(int)
        pred_binary = (predictions > 0.5).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(labels_binary, pred_binary, zero_division=0)
        recall = recall_score(labels_binary, pred_binary, zero_division=0)
        f1 = f1_score(labels_binary, pred_binary, zero_division=0)
        
        try:
            auc = roc_auc_score(labels_binary, predictions)
        except ValueError:
            auc = 0.0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        })
        
        logger.info(f"Evaluation results: {metrics}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model and artifacts
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model and feature extractor must be available to save")
        
        # Save model state
        model_path = os.path.join(self.models_dir, f"{filepath}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save feature extractor
        extractor_path = os.path.join(self.artifacts_dir, f"{filepath}_extractor.pkl")
        self.feature_extractor.save(extractor_path)
        
        # Save training history
        history_path = os.path.join(self.artifacts_dir, f"{filepath}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save model configuration
        config_path = os.path.join(self.artifacts_dir, f"{filepath}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def _save_best_model(self):
        """Save the best model during training"""
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(self.models_dir, 'best_convfm_model.pt'))
    
    def load_model(self, filepath: str):
        """
        Load a trained model and artifacts
        
        Args:
            filepath: Path to the saved model
        """
        # Load model state
        model_path = os.path.join(self.models_dir, f"{filepath}.pt")
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load feature extractor
        extractor_path = os.path.join(self.artifacts_dir, f"{filepath}_extractor.pkl")
        self.feature_extractor = JobFeatureExtractor.load(extractor_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves
        
        Args:
            save_path: Path to save the plot
        """
        if not self.training_history['epochs']:
            logger.warning("No training history available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.training_history['epochs'], self.training_history['train_loss'], 
                label='Train Loss', marker='o')
        ax1.plot(self.training_history['epochs'], self.training_history['val_loss'], 
                label='Validation Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.training_history['epochs'], self.training_history['val_accuracy'], 
                label='Validation Accuracy', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def generate_recommendations(self, user_id: int, job_ids: List[int], 
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Generate job recommendations for a user
        
        Args:
            user_id: User ID
            job_ids: List of job IDs to score
            top_k: Number of top recommendations to return
            
        Returns:
            List of (job_id, score) tuples
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model must be loaded before generating recommendations")
        
        self.model.eval()
        
        # Get user features (simplified - would need user profile data)
        user_skills = []  # Would need actual user skill data
        location_id = 0   # Would need actual user location
        company_id = 0    # Would need actual user company preference
        
        # Get predictions
        scores = self.model.predict_job_scores(
            user_id, job_ids, user_skills, location_id, company_id
        )
        
        # Sort by score and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_scores[:top_k]
    
    def run_full_pipeline(self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame,
                         test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Args:
            jobs_df: Job data DataFrame
            interactions_df: User-job interactions DataFrame
            test_df: Optional test data DataFrame
            
        Returns:
            Pipeline results dictionary
        """
        logger.info("Starting full training pipeline...")
        
        # Prepare data
        train_loader, val_loader, feature_extractor = self.prepare_data(jobs_df, interactions_df)
        
        # Initialize model
        self.initialize_model()
        
        # Train model
        training_history = self.train(train_loader, val_loader)
        
        # Evaluate on validation set
        val_metrics = self.evaluate_model(val_loader)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_model(f"convfm_model_{timestamp}")
        
        # Plot training curves
        curves_path = os.path.join(self.artifacts_dir, f"training_curves_{timestamp}.png")
        self.plot_training_curves(curves_path)
        
        # Prepare results
        results = {
            'training_history': training_history,
            'validation_metrics': val_metrics,
            'model_path': os.path.join(self.models_dir, f"convfm_model_{timestamp}.pt"),
            'feature_extractor_path': os.path.join(self.artifacts_dir, f"convfm_model_{timestamp}_extractor.pkl"),
            'training_curves_path': curves_path,
            'config': self.config
        }
        
        logger.info("Full training pipeline completed successfully")
        
        return results


def create_training_config(
    embedding_dim: int = 64,
    conv_filters: int = 64,
    conv_kernel_size: int = 3,
    dropout_rate: float = 0.2,
    hidden_dims: List[int] = [128, 64, 32],
    learning_rate: float = 0.001,
    batch_size: int = 32,
    num_epochs: int = 50,
    patience: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    models_dir: str = 'models',
    artifacts_dir: str = 'artifacts',
    logs_dir: str = 'logs'
) -> Dict[str, Any]:
    """
    Create a training configuration dictionary
    
    Returns:
        Configuration dictionary
    """
    return {
        'embedding_dim': embedding_dim,
        'conv_filters': conv_filters,
        'conv_kernel_size': conv_kernel_size,
        'dropout_rate': dropout_rate,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'patience': patience,
        'test_size': test_size,
        'random_state': random_state,
        'models_dir': models_dir,
        'artifacts_dir': artifacts_dir,
        'logs_dir': logs_dir
    }
