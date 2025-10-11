"""
Feature Extraction Pipeline for Job and User Data
Implements Phase 1 (Data Processing) from the workflow diagram
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from bs4 import BeautifulSoup


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing for job descriptions and resumes
    """
    
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.max_vocab_size = 50000
        self.max_seq_length = 512
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(str(text), 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common job posting noise
        text = re.sub(r'(apply now|click here|visit our website)', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        text = self.clean_text(text)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from training texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Keep only words above minimum frequency
        filtered_words = [
            word for word, count in word_counts.items() 
            if count >= min_freq
        ]
        
        # Sort by frequency and limit vocabulary size
        filtered_words = sorted(
            filtered_words,
            key=lambda x: word_counts[x],
            reverse=True
        )[:self.max_vocab_size - 2]  # -2 for PAD and UNK
        
        # Build mappings
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(filtered_words, start=2):
            self.vocab[word] = idx
        
        self.word_to_idx = self.vocab
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
    def text_to_sequence(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to sequence of indices"""
        if max_length is None:
            max_length = self.max_seq_length
            
        tokens = self.tokenize(text)
        
        # Convert tokens to indices
        sequence = [
            self.vocab.get(token, self.vocab['<UNK>'])
            for token in tokens
        ]
        
        # Pad or truncate
        if len(sequence) < max_length:
            sequence = sequence + [self.vocab['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence
    
    def batch_texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """Convert batch of texts to sequences"""
        sequences = [self.text_to_sequence(text) for text in texts]
        return np.array(sequences)
    
    def save(self, filepath: str):
        """Save preprocessor state"""
        state = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.vocab = state['vocab']
        self.word_to_idx = state['word_to_idx']
        self.idx_to_word = state['idx_to_word']
        self.max_vocab_size = state['max_vocab_size']
        self.max_seq_length = state['max_seq_length']
        
        print(f"Preprocessor loaded from {filepath}")


class FeatureExtractor:
    """
    Extract and encode features from job postings and user profiles
    """
    
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.encoders = {}
        self.scaler = StandardScaler()
        
        # Initialize label encoders for categorical features
        self.categorical_features = [
            'job_type', 'location', 'experience_level',
            'education_level', 'industry', 'company_size'
        ]
        
        for feature in self.categorical_features:
            self.encoders[feature] = LabelEncoder()
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from job description or resume"""
        # Common tech skills (expand this list based on your domain)
        skill_keywords = {
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php',
            'ruby', 'swift', 'kotlin', 'typescript', 'scala', 'r',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'fastapi',
            'spring', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
            'elasticsearch', 'dynamodb', 'oracle',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'ci/cd', 'git', 'github', 'gitlab',
            
            # Data & ML
            'machine learning', 'deep learning', 'data science', 'nlp',
            'computer vision', 'big data', 'hadoop', 'spark', 'airflow',
            
            # Others
            'agile', 'scrum', 'rest', 'api', 'microservices', 'linux',
            'testing', 'security', 'blockchain', 'iot'
        }
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_job_features(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from a job posting
        
        Args:
            job: Dictionary containing job information
        
        Returns:
            features: Dictionary of extracted features
        """
        features = {}
        
        # Text features
        description = job.get('description', '')
        title = job.get('title', '')
        combined_text = f"{title}. {description}"
        
        features['text'] = combined_text
        features['title'] = title
        
        # Categorical features
        features['job_type'] = job.get('job_type', 'unknown')
        features['location'] = job.get('location', 'unknown')
        features['industry'] = job.get('company_industry', 'unknown')
        
        # Derived features
        features['is_remote'] = 1 if job.get('is_remote', False) else 0
        
        # Experience level (derive from description or job_level)
        job_level = job.get('job_level', '').lower()
        if 'senior' in job_level or 'lead' in job_level:
            features['experience_level'] = 'senior'
        elif 'mid' in job_level or 'intermediate' in job_level:
            features['experience_level'] = 'mid'
        elif 'junior' in job_level or 'entry' in job_level:
            features['experience_level'] = 'junior'
        else:
            features['experience_level'] = 'not_specified'
        
        # Salary features (normalized)
        min_salary = job.get('min_amount', 0) or 0
        max_salary = job.get('max_amount', 0) or 0
        features['avg_salary'] = (min_salary + max_salary) / 2 if max_salary > 0 else 0
        features['has_salary'] = 1 if max_salary > 0 else 0
        
        # Skills extraction
        features['skills'] = self.extract_skills(combined_text)
        features['num_skills'] = len(features['skills'])
        
        # Company features
        features['company'] = job.get('company', 'unknown')
        
        # Posting recency (in days)
        date_posted = job.get('date_posted')
        if date_posted:
            from datetime import datetime
            try:
                posted_date = pd.to_datetime(date_posted)
                days_old = (datetime.now() - posted_date).days
                features['days_old'] = min(days_old, 365)  # Cap at 365 days
            except:
                features['days_old'] = 30  # Default
        else:
            features['days_old'] = 30
        
        return features
    
    def extract_user_features(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from user profile/resume
        
        Args:
            user_profile: Dictionary containing user information
        
        Returns:
            features: Dictionary of extracted features
        """
        features = {}
        
        # Text features (resume/bio)
        resume_text = user_profile.get('resume', '')
        bio = user_profile.get('bio', '')
        combined_text = f"{bio}. {resume_text}"
        
        features['text'] = combined_text
        
        # User preferences
        features['preferred_job_type'] = user_profile.get('preferred_job_type', 'fulltime')
        features['preferred_location'] = user_profile.get('preferred_location', 'any')
        features['is_open_to_remote'] = user_profile.get('open_to_remote', True)
        
        # Experience and education
        features['experience_years'] = user_profile.get('experience_years', 0)
        features['education_level'] = user_profile.get('education_level', 'bachelors')
        
        # Skills
        user_skills = user_profile.get('skills', [])
        if isinstance(user_skills, str):
            user_skills = [s.strip() for s in user_skills.split(',')]
        features['skills'] = user_skills
        features['num_skills'] = len(user_skills)
        
        # Salary expectations
        features['expected_min_salary'] = user_profile.get('expected_min_salary', 0)
        features['expected_max_salary'] = user_profile.get('expected_max_salary', 0)
        
        # Industry preference
        features['preferred_industry'] = user_profile.get('preferred_industry', 'any')
        
        return features
    
    def compute_skill_match(
        self,
        user_skills: List[str],
        job_skills: List[str]
    ) -> float:
        """Compute skill match score between user and job"""
        if not user_skills or not job_skills:
            return 0.0
        
        user_skills_set = set(s.lower() for s in user_skills)
        job_skills_set = set(s.lower() for s in job_skills)
        
        intersection = user_skills_set.intersection(job_skills_set)
        union = user_skills_set.union(job_skills_set)
        
        # Jaccard similarity
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def create_interaction_features(
        self,
        user_features: Dict[str, Any],
        job_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create additional interaction features between user and job
        These are numeric features that capture compatibility
        """
        interactions = []
        
        # 1. Skill match score
        skill_match = self.compute_skill_match(
            user_features.get('skills', []),
            job_features.get('skills', [])
        )
        interactions.append(skill_match)
        
        # 2. Salary compatibility
        user_min = user_features.get('expected_min_salary', 0)
        user_max = user_features.get('expected_max_salary', 0)
        job_salary = job_features.get('avg_salary', 0)
        
        if user_min > 0 and user_max > 0 and job_salary > 0:
            salary_match = 1.0 if user_min <= job_salary <= user_max else 0.5
        else:
            salary_match = 0.5  # Neutral if salary not specified
        interactions.append(salary_match)
        
        # 3. Location match
        user_location = str(user_features.get('preferred_location', '')).lower()
        job_location = str(job_features.get('location', '')).lower()
        is_remote = job_features.get('is_remote', 0)
        user_open_remote = user_features.get('is_open_to_remote', False)
        
        if is_remote and user_open_remote:
            location_match = 1.0
        elif user_location in job_location or job_location in user_location:
            location_match = 1.0
        else:
            location_match = 0.3
        interactions.append(location_match)
        
        # 4. Job type match
        user_job_type = str(user_features.get('preferred_job_type', '')).lower()
        job_type = str(job_features.get('job_type', '')).lower()
        job_type_match = 1.0 if user_job_type in job_type else 0.5
        interactions.append(job_type_match)
        
        # 5. Experience level match
        user_exp_years = user_features.get('experience_years', 0)
        job_exp_level = job_features.get('experience_level', 'not_specified')
        
        if job_exp_level == 'senior' and user_exp_years >= 5:
            exp_match = 1.0
        elif job_exp_level == 'mid' and 2 <= user_exp_years < 5:
            exp_match = 1.0
        elif job_exp_level == 'junior' and user_exp_years < 2:
            exp_match = 1.0
        else:
            exp_match = 0.6  # Partial match
        interactions.append(exp_match)
        
        # 6. Job recency score (newer jobs get higher score)
        days_old = job_features.get('days_old', 30)
        recency_score = max(0, 1 - (days_old / 365))
        interactions.append(recency_score)
        
        # 7. Skill count compatibility
        user_skill_count = user_features.get('num_skills', 0)
        job_skill_count = job_features.get('num_skills', 0)
        
        if user_skill_count > 0 and job_skill_count > 0:
            skill_count_ratio = min(user_skill_count, job_skill_count) / max(user_skill_count, job_skill_count)
        else:
            skill_count_ratio = 0.5
        interactions.append(skill_count_ratio)
        
        # 8-20: Additional placeholder features (can be expanded)
        # Add more domain-specific features as needed
        for i in range(13):
            interactions.append(0.0)
        
        return np.array(interactions[:50])  # Fixed size: 50 features
    
    def fit_encoders(self, jobs_df: pd.DataFrame):
        """Fit label encoders on job data"""
        print("Fitting categorical encoders...")
        
        for feature in self.categorical_features:
            if feature in jobs_df.columns:
                # Handle missing values
                values = jobs_df[feature].fillna('unknown').astype(str)
                self.encoders[feature].fit(values)
                print(f"  {feature}: {len(self.encoders[feature].classes_)} categories")
    
    def encode_categorical(self, value: str, feature_name: str) -> int:
        """Encode a categorical value"""
        if feature_name not in self.encoders:
            return 0
        
        try:
            # Handle unknown categories
            if value not in self.encoders[feature_name].classes_:
                value = 'unknown'
            return self.encoders[feature_name].transform([value])[0]
        except:
            return 0
    
    def prepare_batch_data(
        self,
        user_features: Dict[str, Any],
        job_features_list: List[Dict[str, Any]],
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch data for model input
        
        Returns:
            batch_data: Dictionary containing tensors for model
        """
        batch_size = len(job_features_list)
        
        # Process user text (repeated for all jobs)
        user_text = self.text_preprocessor.text_to_sequence(user_features['text'])
        user_text_batch = np.tile(user_text, (batch_size, 1))
        
        # Process job texts
        job_texts = []
        for job_feat in job_features_list:
            job_text = self.text_preprocessor.text_to_sequence(job_feat['text'])
            job_texts.append(job_text)
        job_texts_batch = np.array(job_texts)
        
        # Process categorical features for user
        user_categorical = {}
        for feat in ['preferred_job_type', 'preferred_location', 'education_level']:
            if feat in user_features:
                # Map to model's expected keys
                model_key = feat.replace('preferred_', '')
                encoded = self.encode_categorical(
                    str(user_features[feat]),
                    model_key if model_key in self.encoders else feat
                )
                user_categorical[model_key] = torch.tensor([encoded] * batch_size).to(device)
        
        # Process categorical features for jobs
        job_categoricals = []
        for job_feat in job_features_list:
            job_cat = {}
            for feat in ['job_type', 'location', 'experience_level', 'industry']:
                if feat in job_feat:
                    encoded = self.encode_categorical(str(job_feat[feat]), feat)
                    job_cat[feat] = torch.tensor([encoded]).to(device)
            job_categoricals.append(job_cat)
        
        # Create interaction features
        additional_features = []
        for job_feat in job_features_list:
            interaction = self.create_interaction_features(user_features, job_feat)
            additional_features.append(interaction)
        additional_features_batch = np.array(additional_features)
        
        # Convert to tensors
        batch_data = {
            'user_text': torch.tensor(user_text_batch, dtype=torch.long).to(device),
            'job_text': torch.tensor(job_texts_batch, dtype=torch.long).to(device),
            'user_categorical': user_categorical,
            'job_categoricals': job_categoricals,
            'additional_features': torch.tensor(additional_features_batch, dtype=torch.float32).to(device)
        }
        
        return batch_data
    
    def save(self, filepath: str):
        """Save feature extractor state"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'encoders': self.encoders,
            'categorical_features': self.categorical_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save text preprocessor separately
        preprocessor_path = filepath.replace('.pkl', '_preprocessor.pkl')
        self.text_preprocessor.save(preprocessor_path)
        
        print(f"Feature extractor saved to {filepath}")
    
    def load(self, filepath: str):
        """Load feature extractor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.encoders = state['encoders']
        self.categorical_features = state['categorical_features']
        
        # Load text preprocessor
        preprocessor_path = filepath.replace('.pkl', '_preprocessor.pkl')
        self.text_preprocessor.load(preprocessor_path)
        
        print(f"Feature extractor loaded from {filepath}")


class DatasetBuilder:
    """
    Build training dataset from scraped jobs and user interactions
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
    
    def create_training_data(
        self,
        jobs_df: pd.DataFrame,
        user_interactions: pd.DataFrame,
        user_profiles: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[float]]:
        """
        Create training data from jobs and user interactions
        
        Args:
            jobs_df: DataFrame of job postings
            user_interactions: DataFrame with columns [user_id, job_id, rating]
                              rating: 1 (applied/saved), 0 (ignored), -1 (rejected)
            user_profiles: Dictionary mapping user_id to profile dict
        
        Returns:
            training_samples: List of feature dictionaries
            labels: List of ratings
        """
        print("Creating training dataset...")
        
        training_samples = []
        labels = []
        
        # Group interactions by user
        for user_id in user_interactions['user_id'].unique():
            if user_id not in user_profiles:
                continue
            
            user_data = user_interactions[user_interactions['user_id'] == user_id]
            user_profile = user_profiles[user_id]
            user_features = self.feature_extractor.extract_user_features(user_profile)
            
            for _, interaction in user_data.iterrows():
                job_id = interaction['job_id']
                rating = interaction['rating']
                
                # Find job in jobs_df
                job_row = jobs_df[jobs_df['id'] == job_id]
                if job_row.empty:
                    continue
                
                job_dict = job_row.iloc[0].to_dict()
                job_features = self.feature_extractor.extract_job_features(job_dict)
                
                # Create sample
                sample = {
                    'user_features': user_features,
                    'job_features': job_features,
                    'user_id': user_id,
                    'job_id': job_id
                }
                
                training_samples.append(sample)
                labels.append(rating)
        
        print(f"Created {len(training_samples)} training samples")
        return training_samples, labels
    
    def create_implicit_feedback_data(
        self,
        jobs_df: pd.DataFrame,
        user_profiles: Dict[str, Dict],
        positive_samples_per_user: int = 10,
        negative_samples_per_user: int = 20
    ) -> Tuple[List[Dict], List[float]]:
        """
        Create training data using implicit feedback
        (when explicit ratings are not available)
        
        This simulates user preferences based on:
        - Skill matching
        - Salary compatibility
        - Location preferences
        """
        print("Creating implicit feedback dataset...")
        
        training_samples = []
        labels = []
        
        for user_id, user_profile in user_profiles.items():
            user_features = self.feature_extractor.extract_user_features(user_profile)
            
            # Compute match scores for all jobs
            job_scores = []
            for _, job_row in jobs_df.iterrows():
                job_dict = job_row.to_dict()
                job_features = self.feature_extractor.extract_job_features(job_dict)
                
                # Compute compatibility score
                skill_match = self.feature_extractor.compute_skill_match(
                    user_features.get('skills', []),
                    job_features.get('skills', [])
                )
                
                job_scores.append({
                    'job_dict': job_dict,
                    'job_features': job_features,
                    'score': skill_match
                })
            
            # Sort jobs by score
            job_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Select positive samples (high match)
            for job_data in job_scores[:positive_samples_per_user]:
                sample = {
                    'user_features': user_features,
                    'job_features': job_data['job_features'],
                    'user_id': user_id,
                    'job_id': job_data['job_dict'].get('id', 'unknown')
                }
                training_samples.append(sample)
                labels.append(1.0)  # Positive label
            
            # Select negative samples (low match)
            for job_data in job_scores[-negative_samples_per_user:]:
                sample = {
                    'user_features': user_features,
                    'job_features': job_data['job_features'],
                    'user_id': user_id,
                    'job_id': job_data['job_dict'].get('id', 'unknown')
                }
                training_samples.append(sample)
                labels.append(0.0)  # Negative label
        
        print(f"Created {len(training_samples)} implicit feedback samples")
        return training_samples, labels