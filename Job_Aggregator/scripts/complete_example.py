#!/usr/bin/env python3
"""
Complete Example: End-to-End Job Recommendation System

This script demonstrates the complete workflow from data collection to recommendation generation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from jobspy import scrape_jobs
from ml_models.training_pipeline import ConvFMTrainingPipeline, create_training_config
from ml_models.convfm_model import ConvFMJobRecommender
from ml_models.feature_extractor import JobFeatureExtractor
from api.recommendation_api import RecommendationAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobRecommendationSystem:
    """
    Complete job recommendation system that handles:
    1. Data collection
    2. Model training
    3. Recommendation generation
    4. API serving
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        self.jobs_df = None
        self.interactions_df = None
        self.feature_extractor = None
        self.model = None
        self.api = None
    
    def collect_job_data(self, keywords: list, locations: list, num_pages: int = 2) -> pd.DataFrame:
        """
        Collect job data using the jobspy scraper
        
        Args:
            keywords: List of job keywords to search for
            locations: List of locations to search in
            num_pages: Number of pages to scrape per search
            
        Returns:
            DataFrame with collected job data
        """
        logger.info("Starting job data collection...")
        
        all_jobs = []
        
        for keyword in keywords:
            for location in locations:
                logger.info(f"Scraping jobs for '{keyword}' in '{location}'...")
                
                try:
                    jobs = scrape_jobs(
                        site_name=["indeed", "linkedin", "glassdoor"],
                        search_term=keyword,
                        location=location,
                        results_wanted=num_pages * 15,  # ~15 jobs per page
                        country_indeed="USA"
                    )
                    
                    if len(jobs) > 0:
                        # Add search metadata
                        jobs['search_keyword'] = keyword
                        jobs['search_location'] = location
                        jobs['collected_at'] = datetime.now().isoformat()
                        
                        all_jobs.append(jobs)
                        logger.info(f"Collected {len(jobs)} jobs for '{keyword}' in '{location}'")
                    else:
                        logger.warning(f"No jobs found for '{keyword}' in '{location}'")
                        
                except Exception as e:
                    logger.error(f"Error scraping jobs for '{keyword}' in '{location}': {str(e)}")
                    continue
        
        if not all_jobs:
            logger.error("No jobs collected. Check your search terms and locations.")
            return pd.DataFrame()
        
        # Combine all job data
        self.jobs_df = pd.concat(all_jobs, ignore_index=True)
        
        # Clean and standardize data
        self.jobs_df = self._clean_job_data(self.jobs_df)
        
        # Save collected data
        jobs_file = os.path.join(self.data_dir, "collected_jobs.csv")
        self.jobs_df.to_csv(jobs_file, index=False)
        logger.info(f"Saved {len(self.jobs_df)} jobs to {jobs_file}")
        
        return self.jobs_df
    
    def _clean_job_data(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize job data"""
        logger.info("Cleaning job data...")
        
        # Remove duplicates
        initial_count = len(jobs_df)
        jobs_df = jobs_df.drop_duplicates(subset=['job_url'], keep='first')
        logger.info(f"Removed {initial_count - len(jobs_df)} duplicate jobs")
        
        # Add job_id if not present
        if 'job_id' not in jobs_df.columns:
            jobs_df['job_id'] = range(len(jobs_df))
        
        # Standardize job types
        job_type_mapping = {
            'full-time': 'fulltime',
            'full time': 'fulltime',
            'part-time': 'parttime',
            'part time': 'parttime',
            'contract': 'contract',
            'internship': 'internship',
            'temporary': 'temporary'
        }
        
        if 'job_type' in jobs_df.columns:
            jobs_df['job_type'] = jobs_df['job_type'].str.lower().map(job_type_mapping).fillna('fulltime')
        else:
            jobs_df['job_type'] = 'fulltime'
        
        # Standardize salary data
        if 'salary' in jobs_df.columns:
            jobs_df['salary'] = pd.to_numeric(jobs_df['salary'], errors='coerce')
        else:
            jobs_df['salary'] = np.nan
        
        # Standardize remote work
        if 'is_remote' not in jobs_df.columns:
            # Infer from description
            jobs_df['is_remote'] = jobs_df['description'].str.contains(
                'remote|work from home|wfh', case=False, na=False
            )
        
        # Fill missing values
        jobs_df['company'] = jobs_df['company'].fillna('Unknown')
        jobs_df['location'] = jobs_df['location'].fillna('Unknown')
        jobs_df['description'] = jobs_df['description'].fillna('')
        
        # Add experience level inference
        jobs_df['experience_level'] = self._infer_experience_level(jobs_df)
        
        logger.info("Job data cleaning completed")
        return jobs_df
    
    def _infer_experience_level(self, jobs_df: pd.DataFrame) -> pd.Series:
        """Infer experience level from job title and description"""
        def get_experience_level(title, description):
            text = f"{title} {description}".lower()
            
            if any(word in text for word in ['senior', 'lead', 'principal', 'architect', 'manager']):
                return 'senior'
            elif any(word in text for word in ['junior', 'entry', 'associate', 'trainee']):
                return 'junior'
            elif any(word in text for word in ['intern', 'internship', 'student']):
                return 'entry'
            else:
                return 'mid'
        
        return jobs_df.apply(lambda row: get_experience_level(row.get('title', ''), row.get('description', '')), axis=1)
    
    def create_user_interactions(self, num_users: int = 500, interactions_per_user: int = 10) -> pd.DataFrame:
        """
        Create realistic user interactions for training
        
        Args:
            num_users: Number of users to create
            interactions_per_user: Average interactions per user
            
        Returns:
            DataFrame with user interactions
        """
        logger.info(f"Creating interactions for {num_users} users...")
        
        if self.jobs_df is None or len(self.jobs_df) == 0:
            raise ValueError("No job data available. Run collect_job_data() first.")
        
        interactions = []
        
        for user_id in range(num_users):
            # Create user profile
            user_profile = self._create_user_profile(user_id)
            
            # Generate interactions based on user preferences
            user_interactions = self._generate_user_interactions(user_profile, interactions_per_user)
            interactions.extend(user_interactions)
        
        self.interactions_df = pd.DataFrame(interactions)
        
        # Save interactions
        interactions_file = os.path.join(self.data_dir, "user_interactions.csv")
        self.interactions_df.to_csv(interactions_file, index=False)
        logger.info(f"Created {len(self.interactions_df)} interactions for {num_users} users")
        
        return self.interactions_df
    
    def _create_user_profile(self, user_id: int) -> dict:
        """Create a realistic user profile"""
        # Simulate user preferences
        preferred_locations = np.random.choice(
            ['New York', 'San Francisco', 'Austin', 'Seattle', 'Boston', 'Remote'],
            size=np.random.randint(1, 4), replace=False
        )
        
        preferred_companies = np.random.choice(
            self.jobs_df['company'].unique(),
            size=np.random.randint(1, 6), replace=False
        )
        
        salary_expectation = np.random.normal(80000, 25000)  # Mean $80k, std $25k
        salary_expectation = max(40000, salary_expectation)  # Minimum $40k
        
        return {
            'user_id': user_id,
            'preferred_locations': preferred_locations.tolist(),
            'preferred_companies': preferred_companies.tolist(),
            'salary_expectation': salary_expectation,
            'prefers_remote': np.random.choice([True, False], p=[0.3, 0.7])
        }
    
    def _generate_user_interactions(self, user_profile: dict, num_interactions: int) -> list:
        """Generate realistic user interactions based on profile"""
        interactions = []
        
        # Filter jobs based on user preferences
        candidate_jobs = self.jobs_df.copy()
        
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
        
        # Salary preference
        salary_mask = candidate_jobs['salary'].fillna(user_profile['salary_expectation']) >= user_profile['salary_expectation'] * 0.8
        candidate_jobs = candidate_jobs[salary_mask]
        
        if len(candidate_jobs) == 0:
            # If no jobs match preferences, use random selection
            candidate_jobs = self.jobs_df
        
        # Select interactions
        num_interactions = min(num_interactions, len(candidate_jobs))
        selected_jobs = candidate_jobs.sample(n=num_interactions, replace=False)
        
        for _, job in selected_jobs.iterrows():
            # Calculate realistic rating based on preferences
            rating = self._calculate_job_rating(job, user_profile)
            
            interactions.append({
                'user_id': user_profile['user_id'],
                'job_id': job['job_id'],
                'rating': rating,
                'timestamp': datetime.now().isoformat(),
                'interaction_type': 'view'  # Could be 'apply', 'save', 'view', etc.
            })
        
        return interactions
    
    def _calculate_job_rating(self, job: pd.Series, user_profile: dict) -> float:
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
    
    def train_model(self, epochs: int = 30, batch_size: int = 32) -> dict:
        """
        Train the ConvFM recommendation model
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        if self.jobs_df is None or self.interactions_df is None:
            raise ValueError("No training data available. Run data collection first.")
        
        # Create training configuration
        config = create_training_config(
            embedding_dim=64,
            conv_filters=64,
            dropout_rate=0.2,
            hidden_dims=[128, 64, 32],
            learning_rate=0.001,
            batch_size=batch_size,
            num_epochs=epochs,
            patience=10,
            test_size=0.2,
            random_state=42,
            models_dir=self.models_dir,
            artifacts_dir='artifacts',
            logs_dir='logs'
        )
        
        # Initialize training pipeline
        pipeline = ConvFMTrainingPipeline(config)
        
        # Run training
        results = pipeline.run_full_pipeline(self.jobs_df, self.interactions_df)
        
        # Load trained model
        self.model = pipeline.model
        self.feature_extractor = pipeline.feature_extractor
        
        logger.info("Model training completed successfully!")
        return results
    
    def generate_recommendations(self, user_id: int, top_k: int = 10) -> list:
        """
        Generate job recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of top recommendations
            
        Returns:
            List of job recommendations
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model not trained. Run train_model() first.")
        
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Get all job IDs
        job_ids = self.jobs_df['job_id'].tolist()
        
        # Generate user profile (simplified)
        user_skills = []  # Would need actual user skill data
        location_id = 0   # Would need actual user location
        company_id = 0    # Would need actual user company preference
        
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
        top_job_ids = [job_id for job_id, score in sorted_scores[:top_k]]
        
        # Format recommendations
        recommendations = []
        for job_id, score in sorted_scores[:top_k]:
            job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
            
            recommendation = {
                'job_id': int(job_id),
                'title': job_data['title'],
                'company': job_data['company'],
                'location': job_data['location'],
                'score': float(score),
                'salary': float(job_data.get('salary', 0)) if pd.notna(job_data.get('salary')) else None,
                'job_type': job_data.get('job_type'),
                'is_remote': bool(job_data.get('is_remote', False)),
                'url': job_data.get('job_url'),
                'description': str(job_data.get('description', ''))[:200] + "..." if len(str(job_data.get('description', ''))) > 200 else str(job_data.get('description', ''))
            }
            
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8001):
        """
        Start the recommendation API server
        
        Args:
            host: Server host
            port: Server port
        """
        logger.info(f"Starting API server on {host}:{port}")
        
        # Find the best model
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pt')]
        if not model_files:
            raise ValueError("No trained model found. Run train_model() first.")
        
        best_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.models_dir, x)))
        model_path = os.path.join(self.models_dir, best_model)
        
        # Find corresponding config file
        config_files = [f for f in os.listdir('artifacts') if f.endswith('_config.json')]
        if config_files:
            config_path = os.path.join('artifacts', config_files[0])
        else:
            config_path = None
        
        # Create and start API
        self.api = RecommendationAPI(model_path, config_path)
        
        import uvicorn
        uvicorn.run(
            self.api.app,
            host=host,
            port=port,
            log_level="info"
        )


def main():
    """Main function demonstrating the complete workflow"""
    logger.info("Starting complete job recommendation system example...")
    
    # Initialize system
    system = JobRecommendationSystem()
    
    try:
        # Step 1: Collect job data
        logger.info("Step 1: Collecting job data...")
        keywords = ['python developer', 'data scientist', 'machine learning engineer', 'software engineer']
        locations = ['New York', 'San Francisco', 'Austin', 'Remote']
        
        jobs_df = system.collect_job_data(keywords, locations, num_pages=2)
        
        if len(jobs_df) == 0:
            logger.error("No jobs collected. Exiting.")
            return 1
        
        # Step 2: Create user interactions
        logger.info("Step 2: Creating user interactions...")
        interactions_df = system.create_user_interactions(num_users=200, interactions_per_user=8)
        
        # Step 3: Train model
        logger.info("Step 3: Training recommendation model...")
        training_results = system.train_model(epochs=20, batch_size=16)
        
        # Step 4: Generate recommendations
        logger.info("Step 4: Generating recommendations...")
        for user_id in [0, 1, 2]:
            recommendations = system.generate_recommendations(user_id, top_k=5)
            logger.info(f"User {user_id} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec['title']} at {rec['company']} (Score: {rec['score']:.3f})")
        
        # Step 5: Start API server (optional)
        logger.info("Step 5: API server ready. To start:")
        logger.info("  python scripts/complete_example.py --start-api")
        
        return 0
        
    except Exception as e:
        logger.error(f"System failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Job Recommendation System')
    parser.add_argument('--start-api', action='store_true',
                       help='Start the API server after training')
    parser.add_argument('--api-host', type=str, default='0.0.0.0',
                       help='API server host')
    parser.add_argument('--api-port', type=int, default=8001,
                       help='API server port')
    
    args = parser.parse_args()
    
    if args.start_api:
        # Just start the API server
        system = JobRecommendationSystem()
        system.start_api_server(args.api_host, args.api_port)
    else:
        # Run the complete example
        exit_code = main()
        sys.exit(exit_code)
