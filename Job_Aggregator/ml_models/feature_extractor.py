"""
Feature Extractor for Job Recommendation

This module handles feature extraction and preprocessing for job recommendation data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class JobFeatureExtractor:
    """
    Feature extractor for job recommendation system
    """
    
    def __init__(self, max_skills: int = 50, max_text_features: int = 1000):
        self.max_skills = max_skills
        self.max_text_features = max_text_features
        
        # Encoders and scalers
        self.user_encoder = LabelEncoder()
        self.job_encoder = LabelEncoder()
        self.skill_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.company_encoder = LabelEncoder()
        
        self.salary_scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Feature mappings
        self.skill_vocab = {}
        self.feature_stats = {}
        
        self.is_fitted = False
    
    def fit(self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame) -> 'JobFeatureExtractor':
        """
        Fit the feature extractor on training data
        
        Args:
            jobs_df: DataFrame with job information
            interactions_df: DataFrame with user-job interactions
            
        Returns:
            Fitted feature extractor
        """
        logger.info("Fitting feature extractor...")
        
        # Fit label encoders
        self.user_encoder.fit(interactions_df['user_id'].unique())
        self.job_encoder.fit(jobs_df['job_id'].unique())
        self.location_encoder.fit(jobs_df['location'].fillna('Unknown').unique())
        self.company_encoder.fit(jobs_df['company'].fillna('Unknown').unique())
        
        # Extract and fit skills
        all_skills = self._extract_skills(jobs_df)
        self.skill_encoder.fit(all_skills)
        
        # Fit text vectorizer on job descriptions
        job_descriptions = jobs_df['description'].fillna('').astype(str)
        self.text_vectorizer.fit(job_descriptions)
        
        # Fit salary scaler
        salary_data = jobs_df['salary'].dropna()
        if len(salary_data) > 0:
            self.salary_scaler.fit(salary_data.values.reshape(-1, 1))
        
        # Build skill vocabulary
        self._build_skill_vocab(jobs_df)
        
        # Calculate feature statistics
        self._calculate_feature_stats(jobs_df, interactions_df)
        
        self.is_fitted = True
        logger.info("Feature extractor fitted successfully")
        
        return self
    
    def transform_jobs(self, jobs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform job data into features
        
        Args:
            jobs_df: DataFrame with job information
            
        Returns:
            Dictionary of feature arrays
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        logger.info("Transforming job features...")
        
        # Basic features
        job_ids = self.job_encoder.transform(jobs_df['job_id'])
        locations = self.location_encoder.transform(jobs_df['location'].fillna('Unknown'))
        companies = self.company_encoder.transform(jobs_df['company'].fillna('Unknown'))
        
        # Salary features
        salaries = jobs_df['salary'].fillna(jobs_df['salary'].median())
        if len(salaries) > 0:
            salaries_scaled = self.salary_scaler.transform(salaries.values.reshape(-1, 1)).flatten()
        else:
            salaries_scaled = np.zeros(len(jobs_df))
        
        # Skill features
        skill_features = self._extract_job_skills(jobs_df)
        
        # Text features
        descriptions = jobs_df['description'].fillna('').astype(str)
        text_features = self.text_vectorizer.transform(descriptions).toarray()
        
        # Additional features
        job_types = self._encode_job_types(jobs_df)
        experience_levels = self._encode_experience_levels(jobs_df)
        remote_flags = jobs_df['is_remote'].fillna(False).astype(int).values
        
        features = {
            'job_ids': job_ids,
            'locations': locations,
            'companies': companies,
            'salaries': salaries_scaled,
            'skill_features': skill_features,
            'text_features': text_features,
            'job_types': job_types,
            'experience_levels': experience_levels,
            'remote_flags': remote_flags
        }
        
        logger.info(f"Transformed {len(jobs_df)} jobs into features")
        return features
    
    def transform_interactions(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform user-job interaction data
        
        Args:
            interactions_df: DataFrame with user-job interactions
            
        Returns:
            Dictionary of feature arrays
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        logger.info("Transforming interaction features...")
        
        # Basic features
        user_ids = self.user_encoder.transform(interactions_df['user_id'])
        job_ids = self.job_encoder.transform(interactions_df['job_id'])
        
        # Labels
        labels = interactions_df['rating'].values if 'rating' in interactions_df.columns else np.ones(len(interactions_df))
        
        # User features (if available)
        user_features = self._extract_user_features(interactions_df)
        
        features = {
            'user_ids': user_ids,
            'job_ids': job_ids,
            'labels': labels,
            'user_features': user_features
        }
        
        logger.info(f"Transformed {len(interactions_df)} interactions into features")
        return features
    
    def _extract_skills(self, jobs_df: pd.DataFrame) -> List[str]:
        """Extract all unique skills from job data"""
        skills = set()
        
        # Extract from job descriptions
        for desc in jobs_df['description'].fillna(''):
            if isinstance(desc, str):
                desc_skills = self._parse_skills_from_text(desc)
                skills.update(desc_skills)
        
        # Extract from skills column if available
        if 'skills' in jobs_df.columns:
            for skill_list in jobs_df['skills'].fillna(''):
                if isinstance(skill_list, str):
                    skill_items = [s.strip() for s in skill_list.split(',') if s.strip()]
                    skills.update(skill_items)
        
        return list(skills)
    
    def _parse_skills_from_text(self, text: str) -> List[str]:
        """Parse skills from job description text"""
        if not isinstance(text, str):
            return []
        
        # Common technical skills patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Go|Rust|Swift|Kotlin|Scala)\b',
            r'\b(?:React|Angular|Vue|Node\.?js|Django|Flask|Spring|Express)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|MongoDB|PostgreSQL|MySQL)\b',
            r'\b(?:Machine Learning|AI|Deep Learning|TensorFlow|PyTorch|Scikit-learn)\b',
            r'\b(?:Data Science|Analytics|SQL|Pandas|NumPy|R|Tableau|Power BI)\b',
            r'\b(?:DevOps|CI/CD|Linux|Unix|Agile|Scrum|API|REST|GraphQL)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([match.lower() for match in matches])
        
        return list(skills)
    
    def _extract_job_skills(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Extract skill features for jobs"""
        skill_features = np.zeros((len(jobs_df), self.max_skills), dtype=np.int32)
        
        for idx, row in jobs_df.iterrows():
            skills = self._parse_skills_from_text(str(row.get('description', '')))
            
            # Encode skills
            encoded_skills = []
            for skill in skills[:self.max_skills]:
                try:
                    encoded_skill = self.skill_encoder.transform([skill])[0]
                    encoded_skills.append(encoded_skill)
                except ValueError:
                    continue
            
            # Fill skill features
            skill_features[idx, :len(encoded_skills)] = encoded_skills
        
        return skill_features
    
    def _extract_user_features(self, interactions_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract user-specific features"""
        user_features = {}
        
        # User interaction statistics
        user_stats = interactions_df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'job_id': 'nunique'
        }).fillna(0)
        
        user_features['avg_rating'] = user_stats[('rating', 'mean')].values
        user_features['rating_std'] = user_stats[('rating', 'std')].values
        user_features['num_interactions'] = user_stats[('rating', 'count')].values
        user_features['unique_jobs'] = user_stats[('job_id', 'nunique')].values
        
        return user_features
    
    def _encode_job_types(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Encode job types"""
        job_type_mapping = {
            'fulltime': 1,
            'part-time': 2,
            'contract': 3,
            'internship': 4,
            'freelance': 5
        }
        
        job_types = jobs_df['job_type'].fillna('fulltime').map(job_type_mapping).fillna(1).values
        return job_types.astype(np.int32)
    
    def _encode_experience_levels(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Encode experience levels"""
        exp_mapping = {
            'entry': 1,
            'junior': 2,
            'mid': 3,
            'senior': 4,
            'lead': 5,
            'executive': 6
        }
        
        exp_levels = jobs_df['experience_level'].fillna('mid').map(exp_mapping).fillna(3).values
        return exp_levels.astype(np.int32)
    
    def _build_skill_vocab(self, jobs_df: pd.DataFrame):
        """Build skill vocabulary with frequencies"""
        skill_counter = Counter()
        
        for desc in jobs_df['description'].fillna(''):
            if isinstance(desc, str):
                skills = self._parse_skills_from_text(desc)
                skill_counter.update(skills)
        
        self.skill_vocab = dict(skill_counter.most_common(self.max_skills))
    
    def _calculate_feature_stats(self, jobs_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Calculate feature statistics"""
        self.feature_stats = {
            'num_users': len(interactions_df['user_id'].unique()),
            'num_jobs': len(jobs_df),
            'num_skills': len(self.skill_vocab),
            'num_locations': len(jobs_df['location'].unique()),
            'num_companies': len(jobs_df['company'].unique()),
            'avg_salary': jobs_df['salary'].mean() if 'salary' in jobs_df.columns else 0,
            'avg_rating': interactions_df['rating'].mean() if 'rating' in interactions_df.columns else 0
        }
    
    def save(self, filepath: str):
        """Save the fitted feature extractor"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted feature extractor")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'JobFeatureExtractor':
        """Load a fitted feature extractor"""
        with open(filepath, 'rb') as f:
            extractor = pickle.load(f)
        
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor


class JobDataset(Dataset):
    """
    PyTorch Dataset for job recommendation data
    """
    
    def __init__(self, features: Dict[str, np.ndarray], labels: Optional[np.ndarray] = None):
        self.features = features
        self.labels = labels
        self.length = len(next(iter(features.values())))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = {}
        
        for key, values in self.features.items():
            if isinstance(values, np.ndarray):
                item[key] = torch.tensor(values[idx], dtype=torch.long if values.dtype in [np.int32, np.int64] else torch.float32)
            else:
                item[key] = torch.tensor(values[idx])
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return item


def create_dataloader(features: Dict[str, np.ndarray], labels: Optional[np.ndarray] = None,
                     batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader from features
    
    Args:
        features: Dictionary of feature arrays
        labels: Optional labels array
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader
    """
    dataset = JobDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_training_data(jobs_df: pd.DataFrame, interactions_df: pd.DataFrame,
                         test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Prepare training and validation data
    
    Args:
        jobs_df: Job data DataFrame
        interactions_df: User-job interactions DataFrame
        test_size: Fraction of data for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_features, val_features, train_labels, val_labels)
    """
    from sklearn.model_selection import train_test_split
    
    # Fit feature extractor
    extractor = JobFeatureExtractor()
    extractor.fit(jobs_df, interactions_df)
    
    # Transform data
    job_features = extractor.transform_jobs(jobs_df)
    interaction_features = extractor.transform_interactions(interactions_df)
    
    # Split data
    train_indices, val_indices = train_test_split(
        range(len(interactions_df)),
        test_size=test_size,
        random_state=random_state
    )
    
    # Prepare training data
    train_features = {
        'user_ids': interaction_features['user_ids'][train_indices],
        'job_ids': interaction_features['job_ids'][train_indices],
        'skill_ids': job_features['skill_features'][interaction_features['job_ids'][train_indices]],
        'location_ids': job_features['locations'][interaction_features['job_ids'][train_indices]],
        'company_ids': job_features['companies'][interaction_features['job_ids'][train_indices]]
    }
    
    train_labels = interaction_features['labels'][train_indices]
    
    # Prepare validation data
    val_features = {
        'user_ids': interaction_features['user_ids'][val_indices],
        'job_ids': interaction_features['job_ids'][val_indices],
        'skill_ids': job_features['skill_features'][interaction_features['job_ids'][val_indices]],
        'location_ids': job_features['locations'][interaction_features['job_ids'][val_indices]],
        'company_ids': job_features['companies'][interaction_features['job_ids'][val_indices]]
    }
    
    val_labels = interaction_features['labels'][val_indices]
    
    return train_features, val_features, train_labels, val_labels, extractor
