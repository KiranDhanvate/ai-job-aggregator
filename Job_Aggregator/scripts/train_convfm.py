#!/usr/bin/env python3
"""
Training Script for ConvFM Job Recommender

This script trains the ConvFM model for job recommendations using the training pipeline.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.training_pipeline import ConvFMTrainingPipeline, create_training_config
from ml_models.feature_extractor import JobFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str) -> tuple:
    """
    Load training data from CSV files
    
    Args:
        data_dir: Directory containing training data files
        
    Returns:
        Tuple of (jobs_df, interactions_df)
    """
    logger.info(f"Loading data from {data_dir}")
    
    # Load job data
    jobs_file = os.path.join(data_dir, 'collected_jobs.csv')
    if not os.path.exists(jobs_file):
        raise FileNotFoundError(f"Jobs file not found: {jobs_file}")
    
    jobs_df = pd.read_csv(jobs_file)
    logger.info(f"Loaded {len(jobs_df)} jobs")
    
    # Load interaction data
    interactions_file = os.path.join(data_dir, 'user_interactions.csv')
    if not os.path.exists(interactions_file):
        logger.warning(f"Interactions file not found: {interactions_file}")
        logger.info("Creating dummy interactions for training...")
        
        # Create dummy interactions for training
        interactions_df = create_dummy_interactions(jobs_df)
    else:
        interactions_df = pd.read_csv(interactions_file)
        logger.info(f"Loaded {len(interactions_df)} interactions")
    
    return jobs_df, interactions_df


def create_dummy_interactions(jobs_df: pd.DataFrame, num_users: int = 1000) -> pd.DataFrame:
    """
    Create dummy user interactions for training when no real data is available
    
    Args:
        jobs_df: Job data DataFrame
        num_users: Number of dummy users to create
        
    Returns:
        DataFrame with dummy interactions
    """
    import numpy as np
    
    logger.info(f"Creating {num_users} dummy users with interactions")
    
    interactions = []
    
    for user_id in range(num_users):
        # Random number of interactions per user (1-20)
        num_interactions = np.random.randint(1, 21)
        
        # Random job selections
        job_indices = np.random.choice(len(jobs_df), size=num_interactions, replace=False)
        
        for job_idx in job_indices:
            # Random rating based on job features (simplified)
            base_rating = np.random.uniform(0.3, 0.9)
            
            # Add some bias based on job characteristics
            job = jobs_df.iloc[job_idx]
            
            # Higher ratings for remote jobs
            if job.get('is_remote', False):
                base_rating += 0.1
            
            # Higher ratings for certain job types
            if job.get('job_type') in ['fulltime', 'senior']:
                base_rating += 0.05
            
            # Cap rating at 1.0
            rating = min(1.0, base_rating)
            
            interactions.append({
                'user_id': user_id,
                'job_id': job['job_id'],
                'rating': rating,
                'timestamp': datetime.now().isoformat()
            })
    
    interactions_df = pd.DataFrame(interactions)
    logger.info(f"Created {len(interactions_df)} dummy interactions")
    
    return interactions_df


def validate_data(jobs_df: pd.DataFrame, interactions_df: pd.DataFrame) -> bool:
    """
    Validate training data
    
    Args:
        jobs_df: Job data DataFrame
        interactions_df: Interactions DataFrame
        
    Returns:
        True if data is valid, False otherwise
    """
    logger.info("Validating training data...")
    
    # Check required columns
    required_job_columns = ['job_id', 'title', 'company', 'location', 'description']
    missing_job_cols = [col for col in required_job_columns if col not in jobs_df.columns]
    
    if missing_job_cols:
        logger.error(f"Missing required job columns: {missing_job_cols}")
        return False
    
    required_interaction_columns = ['user_id', 'job_id']
    missing_interaction_cols = [col for col in required_interaction_columns if col not in interactions_df.columns]
    
    if missing_interaction_cols:
        logger.error(f"Missing required interaction columns: {missing_interaction_cols}")
        return False
    
    # Check for empty data
    if len(jobs_df) == 0:
        logger.error("No jobs found in dataset")
        return False
    
    if len(interactions_df) == 0:
        logger.error("No interactions found in dataset")
        return False
    
    # Check for overlapping job IDs
    job_ids_in_jobs = set(jobs_df['job_id'].unique())
    job_ids_in_interactions = set(interactions_df['job_id'].unique())
    
    overlapping_jobs = job_ids_in_jobs.intersection(job_ids_in_interactions)
    
    if len(overlapping_jobs) == 0:
        logger.error("No overlapping job IDs between jobs and interactions")
        return False
    
    logger.info(f"Data validation passed. {len(overlapping_jobs)} jobs have interactions")
    return True


def save_training_results(results: dict, output_dir: str):
    """
    Save training results to files
    
    Args:
        results: Training results dictionary
        output_dir: Output directory for results
    """
    logger.info(f"Saving training results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training history
    history_file = os.path.join(output_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump(results['training_history'], f, indent=2)
    
    # Save validation metrics
    metrics_file = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics = {}
        for key, value in results['validation_metrics'].items():
            if hasattr(value, 'tolist'):
                metrics[key] = value.tolist()
            else:
                metrics[key] = value
        json.dump(metrics, f, indent=2)
    
    # Save model configuration
    config_file = os.path.join(output_dir, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(results['config'], f, indent=2)
    
    logger.info("Training results saved successfully")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ConvFM Job Recommender')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing training data files')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for trained models')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--conv_filters', type=int, default=64,
                       help='Number of convolutional filters')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                       help='Hidden layer dimensions')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation data split')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--create_dummy_data', action='store_true',
                       help='Create dummy interaction data if not available')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting ConvFM training pipeline...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load data
        jobs_df, interactions_df = load_data(args.data_dir)
        
        # Validate data
        if not validate_data(jobs_df, interactions_df):
            logger.error("Data validation failed. Exiting.")
            return 1
        
        # Create training configuration
        config = create_training_config(
            embedding_dim=args.embedding_dim,
            conv_filters=args.conv_filters,
            dropout_rate=args.dropout_rate,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            patience=args.patience,
            test_size=args.validation_split,
            random_state=args.random_state,
            models_dir=args.output_dir,
            artifacts_dir='artifacts',
            logs_dir='logs'
        )
        
        # Initialize training pipeline
        pipeline = ConvFMTrainingPipeline(config)
        
        # Run training
        logger.info("Starting model training...")
        results = pipeline.run_full_pipeline(jobs_df, interactions_df)
        
        # Save results
        save_training_results(results, args.output_dir)
        
        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Validation accuracy: {results['validation_metrics']['accuracy']:.4f}")
        logger.info(f"Validation F1-score: {results['validation_metrics']['f1_score']:.4f}")
        logger.info(f"Validation AUC: {results['validation_metrics']['auc']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
