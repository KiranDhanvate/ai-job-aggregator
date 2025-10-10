"""
Training and Evaluation Pipeline for ConvFM Model
Implements Phase 4 (Evaluation) from the workflow diagram
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from convfm_model import ConvFM, ConvFMLoss, create_convfm_model
from feature_extractor import FeatureExtractor, DatasetBuilder


class JobRecommendationDataset(Dataset):
    """
    PyTorch Dataset for job recommendations
    """
    
    def __init__(
        self,
        samples: List[Dict],
        labels: List[float],
        feature_extractor: FeatureExtractor,
        device: str = 'cpu'
    ):
        self.samples = samples
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.device = device
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        user_features = sample['user_features']
        job_features = sample['job_features']
        
        # Convert texts to sequences
        user_text = self.feature_extractor.text_preprocessor.text_to_sequence(
            user_features['text']
        )
        job_text = self.feature_extractor.text_preprocessor.text_to_sequence(
            job_features['text']
        )
        
        # Encode categorical features
        user_categorical = {}
        for feat in ['preferred_job_type', 'preferred_location', 'education_level']:
            if feat in user_features:
                model_key = feat.replace('preferred_', '')
                encoded = self.feature_extractor.encode_categorical(
                    str(user_features[feat]),
                    model_key if model_key in self.feature_extractor.encoders else feat
                )
                user_categorical[model_key] = encoded
        
        job_categorical = {}
        for feat in ['job_type', 'location', 'experience_level', 'industry']:
            if feat in job_features:
                encoded = self.feature_extractor.encode_categorical(
                    str(job_features[feat]), feat
                )
                job_categorical[feat] = encoded
        
        # Create interaction features
        additional_features = self.feature_extractor.create_interaction_features(
            user_features, job_features
        )
        
        return {
            'user_text': torch.tensor(user_text, dtype=torch.long),
            'job_text': torch.tensor(job_text, dtype=torch.long),
            'user_categorical': user_categorical,
            'job_categorical': job_categorical,
            'additional_features': torch.tensor(additional_features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    user_texts = torch.stack([item['user_text'] for item in batch])
    job_texts = torch.stack([item['job_text'] for item in batch])
    labels = torch.stack([item['label'] for item in batch]).unsqueeze(1)
    
    # Collect categorical features
    user_categoricals = {}
    job_categoricals = []
    
    # Get all possible categorical keys
    all_user_keys = set()
    for item in batch:
        all_user_keys.update(item['user_categorical'].keys())
    
    # Stack user categorical features
    for key in all_user_keys:
        values = [item['user_categorical'].get(key, 0) for item in batch]
        user_categoricals[key] = torch.tensor(values, dtype=torch.long)
    
    # Collect job categorical features
    for item in batch:
        job_cat = {k: torch.tensor([v], dtype=torch.long) 
                   for k, v in item['job_categorical'].items()}
        job_categoricals.append(job_cat)
    
    # Stack additional features
    additional_features = torch.stack([item['additional_features'] for item in batch])
    
    return {
        'user_text': user_texts,
        'job_text': job_texts,
        'user_categorical': user_categoricals,
        'job_categoricals': job_categoricals,
        'additional_features': additional_features,
        'labels': labels
    }


class ConvFMTrainer:
    """
    Trainer class for ConvFM model
    """
    
    def __init__(
        self,
        model: ConvFM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = ConvFMLoss(lambda_reg=0.01)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            # Move batch to device
            user_text = batch['user_text'].to(self.device)
            job_text = batch['job_text'].to(self.device)
            additional_features = batch['additional_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Move categorical features to device
            user_categorical = {
                k: v.to(self.device) 
                for k, v in batch['user_categorical'].items()
            }
            job_categoricals = [
                {k: v.to(self.device) for k, v in jc.items()}
                for jc in batch['job_categoricals']
            ]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                user_text,
                job_text,
                user_categorical,
                job_categoricals,
                additional_features
            )
            
            # Compute loss
            loss = self.criterion(predictions, labels, self.model)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
        mae = mean_absolute_error(all_labels, all_predictions)
        
        return avg_loss, rmse, mae
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move batch to device
                user_text = batch['user_text'].to(self.device)
                job_text = batch['job_text'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Move categorical features to device
                user_categorical = {
                    k: v.to(self.device) 
                    for k, v in batch['user_categorical'].items()
                }
                job_categoricals = [
                    {k: v.to(self.device) for k, v in jc.items()}
                    for jc in batch['job_categoricals']
                ]
                
                # Forward pass
                predictions = self.model(
                    user_text,
                    job_text,
                    user_categorical,
                    job_categoricals,
                    additional_features
                )
                
                # Compute loss
                loss = self.criterion(predictions, labels, self.model)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
        mae = mean_absolute_error(all_labels, all_predictions)
        
        return avg_loss, rmse, mae
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: str = './models'
    ):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
        """
        print(f"\n{'='*60}")
        print(f"Starting ConvFM Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_rmse, train_mae = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_rmse, val_mae = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = Path(save_dir) / 'best_convfm_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae
                }, checkpoint_path)
                
                print(f"✅ Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        # Save training history
        history_path = Path(save_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot RMSE
        axes[1].plot(self.history['train_rmse'], label='Train RMSE')
        axes[1].plot(self.history['val_rmse'], label='Val RMSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Squared Error')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot MAE
        axes[2].plot(self.history['train_mae'], label='Train MAE')
        axes[2].plot(self.history['val_mae'], label='Val MAE')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAE')
        axes[2].set_title('Mean Absolute Error')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


def prepare_training_data(
    jobs_csv_path: str,
    feature_extractor: FeatureExtractor,
    val_split: float = 0.2,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare training, validation, and test data loaders
    
    Args:
        jobs_csv_path: Path to CSV file with job postings
        feature_extractor: Feature extractor instance
        val_split: Validation set proportion
        test_split: Test set proportion
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading job data...")
    jobs_df = pd.read_csv(jobs_csv_path)
    
    # Build vocabulary
    print("Building vocabulary from job descriptions...")
    all_texts = jobs_df['description'].fillna('').astype(str).tolist()
    all_texts += jobs_df['title'].fillna('').astype(str).tolist()
    feature_extractor.text_preprocessor.build_vocabulary(all_texts)
    
    # Fit encoders
    feature_extractor.fit_encoders(jobs_df)
    
    # Create sample user profiles (in production, load from database)
    print("Creating sample user profiles...")
    user_profiles = create_sample_user_profiles(num_users=100)
    
    # Build dataset
    dataset_builder = DatasetBuilder(feature_extractor)
    samples, labels = dataset_builder.create_implicit_feedback_data(
        jobs_df,
        user_profiles,
        positive_samples_per_user=15,
        negative_samples_per_user=30
    )
    
    # Split data
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels, test_size=val_split + test_split, random_state=42
    )
    
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels,
        test_size=test_split / (val_split + test_split),
        random_state=42
    )
    
    # Create datasets
    train_dataset = JobRecommendationDataset(train_samples, train_labels, feature_extractor)
    val_dataset = JobRecommendationDataset(val_samples, val_labels, feature_extractor)
    test_dataset = JobRecommendationDataset(test_samples, test_labels, feature_extractor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def create_sample_user_profiles(num_users: int = 100) -> Dict[str, Dict]:
    """Create sample user profiles for training"""
    user_profiles = {}
    
    skills_pool = [
        ['python', 'machine learning', 'tensorflow', 'aws'],
        ['java', 'spring', 'microservices', 'kubernetes'],
        ['javascript', 'react', 'nodejs', 'mongodb'],
        ['sql', 'data analysis', 'tableau', 'excel'],
        ['c++', 'algorithms', 'system design', 'linux'],
    ]
    
    locations = ['New York', 'San Francisco', 'Remote', 'Seattle', 'Austin']
    industries = ['technology', 'finance', 'healthcare', 'e-commerce', 'consulting']
    
    for i in range(num_users):
        user_id = f"user_{i+1}"
        user_profiles[user_id] = {
            'resume': f"Experienced professional with skills in {', '.join(np.random.choice(skills_pool[i % len(skills_pool)], 3, replace=False))}",
            'bio': "Passionate about technology and innovation",
            'skills': list(skills_pool[i % len(skills_pool)]),
            'experience_years': np.random.randint(0, 15),
            'education_level': np.random.choice(['bachelors', 'masters', 'phd']),
            'preferred_job_type': np.random.choice(['fulltime', 'parttime', 'contract']),
            'preferred_location': np.random.choice(locations),
            'open_to_remote': np.random.choice([True, False], p=[0.7, 0.3]),
            'expected_min_salary': np.random.randint(50000, 120000),
            'expected_max_salary': np.random.randint(120000, 200000),
            'preferred_industry': np.random.choice(industries)
        }
    
    return user_profiles