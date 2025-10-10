#!/usr/bin/env python3
"""
Training Script for ConvFM Job Recommender using existing CSV data

This script trains the ConvFM model using your existing jobs_output.csv file.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

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


def load_and_prepare_data(jobs_csv_path: str) -> tuple:
    """
    Load and prepare training data from existing CSV file
    
    Args:
        jobs_csv_path: Path to the jobs CSV file
        
    Returns:
        Tuple of (jobs_df, interactions_df)
    """
    logger.info(f"Loading jobs data from {jobs_csv_path}")
    
    if not os.path.exists(jobs_csv_path):
        raise FileNotFoundError(f"Jobs file not found: {jobs_csv_path}")
    
    # Load job data
    jobs_df = pd.read_csv(jobs_csv_path)
    logger.info(f"Loaded {len(jobs_df)} jobs from CSV")
    
    # Clean and prepare job data
    jobs_df = clean_job_data(jobs_df)
    
    # Create dummy interactions for training
    interactions_df = create_interactions_from_jobs(jobs_df)
    
    return jobs_df, interactions_df


def clean_job_data(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize job data from CSV"""
    logger.info("Cleaning and standardizing job data...")
    
    # Create job_id if not present
    if 'id' in jobs_df.columns:
        jobs_df['job_id'] = jobs_df['id']
    else:
        jobs_df['job_id'] = range(len(jobs_df))
    
    # Handle missing values
    jobs_df['title'] = jobs_df['title'].fillna('Unknown Position')
    jobs_df['company'] = jobs_df['company'].fillna('Unknown Company')
    jobs_df['location'] = jobs_df['location'].fillna('Unknown Location')
    jobs_df['description'] = jobs_df['description'].fillna('')
    
    # Standardize job types
    if 'job_type' in jobs_df.columns:
        job_type_mapping = {
            'fulltime': 'fulltime',
            'full-time': 'fulltime',
            'full time': 'fulltime',
            'part-time': 'parttime',
            'part time': 'parttime',
            'contract': 'contract',
            'internship': 'internship',
            'temporary': 'temporary'
        }
        jobs_df['job_type'] = jobs_df['job_type'].str.lower().map(job_type_mapping).fillna('fulltime')
    else:
        jobs_df['job_type'] = 'fulltime'
    
    # Handle salary data
    if 'min_amount' in jobs_df.columns and 'max_amount' in jobs_df.columns:
        # Create average salary from min and max
        jobs_df['salary'] = (jobs_df['min_amount'].fillna(0) + jobs_df['max_amount'].fillna(0)) / 2
        jobs_df['salary'] = jobs_df['salary'].replace(0, np.nan)
    else:
        jobs_df['salary'] = np.nan
    
    # Handle remote work
    if 'is_remote' in jobs_df.columns:
        jobs_df['is_remote'] = jobs_df['is_remote'].fillna(False)
        # Convert string values to boolean
        jobs_df['is_remote'] = jobs_df['is_remote'].astype(str).str.lower().isin(['true', 'yes', '1', 'remote'])
    else:
        # Infer from description
        jobs_df['is_remote'] = jobs_df['description'].str.contains(
            'remote|work from home|wfh', case=False, na=False
        )
    
    # Add experience level inference
    jobs_df['experience_level'] = infer_experience_level(jobs_df)
    
    # Extract skills from description if skills column is empty
    if 'skills' not in jobs_df.columns or jobs_df['skills'].isna().all():
        logger.info("Extracting skills from job descriptions...")
        jobs_df['skills'] = jobs_df['description'].apply(extract_skills_from_description)
    
    logger.info("Job data cleaning completed")
    return jobs_df


def infer_experience_level(jobs_df: pd.DataFrame) -> pd.Series:
    """Infer experience level from job title and description"""
    def get_experience_level(title, description):
        text = f"{title} {description}".lower()
        
        if any(word in text for word in ['senior', 'lead', 'principal', 'architect', 'manager', 'director']):
            return 'senior'
        elif any(word in text for word in ['junior', 'entry', 'associate', 'trainee']):
            return 'junior'
        elif any(word in text for word in ['intern', 'internship', 'student']):
            return 'entry'
        else:
            return 'mid'
    
    return jobs_df.apply(lambda row: get_experience_level(row.get('title', ''), row.get('description', '')), axis=1)


def extract_skills_from_description(description: str) -> str:
    """Extract skills from job description"""
    if not isinstance(description, str):
        return ''
    
    # Common technical skills patterns
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|C\+\+|C#|Go|Rust|Swift|Kotlin|Scala)\b',
        r'\b(?:React|Angular|Vue|Node\.?js|Django|Flask|Spring|Express)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|MongoDB|PostgreSQL|MySQL)\b',
        r'\b(?:Machine Learning|AI|Deep Learning|TensorFlow|PyTorch|Scikit-learn)\b',
        r'\b(?:Data Science|Analytics|SQL|Pandas|NumPy|R|Tableau|Power BI)\b',
        r'\b(?:DevOps|CI/CD|Linux|Unix|Agile|Scrum|API|REST|GraphQL)\b'
    ]
    
    import re
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        skills.update([match.lower() for match in matches])
    
    return ', '.join(list(skills)[:10])  # Limit to 10 skills


def create_interactions_from_jobs(jobs_df: pd.DataFrame, num_users: int = 1000) -> pd.DataFrame:
    """
    Create realistic user interactions for training from job data
    
    Args:
        jobs_df: Job data DataFrame
        num_users: Number of dummy users to create
        
    Returns:
        DataFrame with user interactions
    """
    logger.info(f"Creating {num_users} dummy users with interactions")
    
    interactions = []
    
    for user_id in range(num_users):
        # Create user profile
        user_profile = create_user_profile(user_id, jobs_df)
        
        # Generate interactions based on user preferences
        num_interactions = np.random.randint(5, 21)
        user_interactions = generate_user_interactions(user_profile, jobs_df, num_interactions)
        interactions.extend(user_interactions)
    
    interactions_df = pd.DataFrame(interactions)
    logger.info(f"Created {len(interactions_df)} interactions for {num_users} users")
    
    return interactions_df


def create_user_profile(user_id: int, jobs_df: pd.DataFrame) -> dict:
    """Create a realistic user profile based on job data"""
    # Get available locations and companies
    available_locations = jobs_df['location'].dropna().unique()
    available_companies = jobs_df['company'].dropna().unique()
    
    # Simulate user preferences
    preferred_locations = np.random.choice(
        available_locations,
        size=min(np.random.randint(1, 4), len(available_locations)),
        replace=False
    )
    
    preferred_companies = np.random.choice(
        available_companies,
        size=min(np.random.randint(1, 6), len(available_companies)),
        replace=False
    )
    
    # Salary expectation based on job data
    if jobs_df['salary'].notna().any():
        avg_salary = jobs_df['salary'].mean()
        salary_expectation = np.random.normal(avg_salary, avg_salary * 0.3)
        salary_expectation = max(avg_salary * 0.5, salary_expectation)
    else:
        salary_expectation = np.random.normal(80000, 25000)
    
    return {
        'user_id': user_id,
        'preferred_locations': preferred_locations.tolist(),
        'preferred_companies': preferred_companies.tolist(),
        'salary_expectation': salary_expectation,
        'prefers_remote': np.random.choice([True, False], p=[0.3, 0.7])
    }


def generate_user_interactions(user_profile: dict, jobs_df: pd.DataFrame, num_interactions: int) -> list:
    """Generate realistic user interactions based on profile"""
    interactions = []
    
    # Filter jobs based on user preferences
    candidate_jobs = jobs_df.copy()
    
    # Location preference
    if user_profile['preferred_locations']:
        location_mask = candidate_jobs['location'].isin(user_profile['preferred_locations'])
        if user_profile['prefers_remote']:
            location_mask = location_mask | candidate_jobs.get('is_remote', False)
        candidate_jobs = candidate_jobs[location_mask]
    
    # Company preference
    if user_profile['preferred_companies']:
        company_mask = candidate_jobs['company'].isin(user_profile['preferred_companies'])
        candidate_jobs = candidate_jobs[company_mask]
    
    # If no jobs match preferences, use all jobs
    if len(candidate_jobs) == 0:
        candidate_jobs = jobs_df
    
    # Select interactions
    num_interactions = min(num_interactions, len(candidate_jobs))
    selected_jobs = candidate_jobs.sample(n=num_interactions, replace=False)
    
    for _, job in selected_jobs.iterrows():
        # Calculate realistic rating based on preferences
        rating = calculate_job_rating(job, user_profile)
        
        interactions.append({
            'user_id': user_profile['user_id'],
            'job_id': job['job_id'],
            'rating': rating,
            'timestamp': datetime.now().isoformat(),
            'interaction_type': 'view'
        })
    
    return interactions


def calculate_job_rating(job: pd.Series, user_profile: dict) -> float:
    """Calculate realistic job rating based on user preferences"""
    rating = 0.5  # Base rating
    
    # Location match
    if job['location'] in user_profile['preferred_locations']:
        rating += 0.2
    elif job.get('is_remote', False) and user_profile['prefers_remote']:
        rating += 0.15
    
    # Company match
    if job['company'] in user_profile['preferred_companies']:
        rating += 0.2
    
    # Salary match
    if pd.notna(job.get('salary')):
        salary_ratio = job['salary'] / user_profile['salary_expectation']
        if salary_ratio >= 1.2:
            rating += 0.1
        elif salary_ratio >= 0.8:
            rating += 0.05
        else:
            rating -= 0.1
    
    # Add some randomness
    rating += np.random.normal(0, 0.1)
    
    # Clamp to [0, 1]
    rating = max(0.0, min(1.0, rating))
    
    return rating


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ConvFM Job Recommender from CSV')
    
    # Data arguments
    parser.add_argument('--jobs-csv', type=str, required=True,
                       help='Path to jobs CSV file')
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Output directory for trained models')
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts',
                       help='Output directory for training artifacts')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--conv-filters', type=int, default=64,
                       help='Number of convolutional filters')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 64, 32],
                       help='Hidden layer dimensions')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation data split')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--num-users', type=int, default=1000,
                       help='Number of dummy users to create')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting ConvFM training from CSV data...")
        logger.info(f"Arguments: {vars(args)}")
        
        # Create output directories
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.artifacts_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load and prepare data
        jobs_df, interactions_df = load_and_prepare_data(args.jobs_csv)
        
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
            models_dir=args.model_dir,
            artifacts_dir=args.artifacts_dir,
            logs_dir='logs'
        )
        
        # Initialize training pipeline
        pipeline = ConvFMTrainingPipeline(config)
        
        # Run training
        logger.info("Starting model training...")
        results = pipeline.run_full_pipeline(jobs_df, interactions_df)
        
        # Print summary
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Validation accuracy: {results['validation_metrics']['accuracy']:.4f}")
        logger.info(f"Validation F1-score: {results['validation_metrics']['f1_score']:.4f}")
        logger.info(f"Validation AUC: {results['validation_metrics']['auc']:.4f}")
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Model Performance:")
        print(f"   • Accuracy: {results['validation_metrics']['accuracy']:.4f}")
        print(f"   • F1-Score: {results['validation_metrics']['f1_score']:.4f}")
        print(f"   • AUC-ROC: {results['validation_metrics']['auc']:.4f}")
        print(f"   • Precision: {results['validation_metrics']['precision']:.4f}")
        print(f"   • Recall: {results['validation_metrics']['recall']:.4f}")
        print(f"\n📁 Files Created:")
        print(f"   • Model: {results['model_path']}")
        print(f"   • Feature Extractor: {results['feature_extractor_path']}")
        print(f"   • Training Curves: {results['training_curves_path']}")
        print("\n🚀 Next Steps:")
        print("   1. Test your model: python scripts/test_recommendations.py")
        print("   2. Start API server: python api/recommendation_api.py")
        print("   3. View training curves in artifacts/ folder")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)