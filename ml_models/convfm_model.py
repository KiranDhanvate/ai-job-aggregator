"""
ConvFM Job Recommender Model

This module implements a Convolutional Factorization Machine (ConvFM) model
for job recommendation based on user profiles and job features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConvFMJobRecommender(nn.Module):
    """
    Convolutional Factorization Machine for Job Recommendation
    
    This model combines:
    - Factorization Machine for feature interactions
    - Convolutional layers for sequence modeling
    - Deep neural network for non-linear feature learning
    """
    
    def __init__(
        self,
        num_users: int,
        num_jobs: int,
        num_skills: int,
        num_locations: int,
        num_companies: int,
        embedding_dim: int = 64,
        conv_filters: int = 64,
        conv_kernel_size: int = 3,
        dropout_rate: float = 0.2,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        super(ConvFMJobRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_jobs = num_jobs
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.location_embedding = nn.Embedding(num_locations, embedding_dim)
        self.company_embedding = nn.Embedding(num_companies, embedding_dim)
        
        # Convolutional layers for sequence modeling
        self.conv1d = nn.Conv1d(embedding_dim, conv_filters, conv_kernel_size, padding=1)
        self.conv_pool = nn.AdaptiveAvgPool1d(1)
        
        # Factorization Machine components
        self.fm_linear = nn.Linear(5 * embedding_dim, 1)  # Linear term
        self.fm_embeddings = nn.ModuleList([
            self.user_embedding,
            self.job_embedding,
            self.skill_embedding,
            self.location_embedding,
            self.company_embedding
        ])
        
        # Deep neural network
        deep_input_dim = 5 * embedding_dim + conv_filters
        self.deep_layers = nn.ModuleList()
        prev_dim = deep_input_dim
        
        for hidden_dim in hidden_dims:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_layers.append(nn.BatchNorm1d(hidden_dim))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final prediction layer
        self.prediction_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, user_ids: torch.Tensor, job_ids: torch.Tensor, 
                skill_ids: torch.Tensor, location_ids: torch.Tensor, 
                company_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvFM model
        
        Args:
            user_ids: User indices [batch_size]
            job_ids: Job indices [batch_size]
            skill_ids: Skill indices [batch_size, max_skills]
            location_ids: Location indices [batch_size]
            company_ids: Company indices [batch_size]
            
        Returns:
            Prediction scores [batch_size, 1]
        """
        batch_size = user_ids.size(0)
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        job_emb = self.job_embedding(job_ids)     # [batch_size, embedding_dim]
        location_emb = self.location_embedding(location_ids)  # [batch_size, embedding_dim]
        company_emb = self.company_embedding(company_ids)     # [batch_size, embedding_dim]
        
        # Handle variable-length skill sequences
        skill_emb = self.skill_embedding(skill_ids)  # [batch_size, max_skills, embedding_dim]
        skill_emb_pooled = torch.mean(skill_emb, dim=1)  # [batch_size, embedding_dim]
        
        # Convolutional feature extraction
        # Combine all embeddings for sequence modeling
        combined_emb = torch.stack([user_emb, job_emb, skill_emb_pooled, 
                                   location_emb, company_emb], dim=2)  # [batch_size, embedding_dim, 5]
        
        # Apply 1D convolution
        conv_out = F.relu(self.conv1d(combined_emb))  # [batch_size, conv_filters, 5]
        conv_pooled = self.conv_pool(conv_out).squeeze(-1)  # [batch_size, conv_filters]
        
        # Factorization Machine components
        embeddings = [user_emb, job_emb, skill_emb_pooled, location_emb, company_emb]
        
        # Linear term
        linear_input = torch.cat(embeddings, dim=1)  # [batch_size, 5 * embedding_dim]
        linear_term = self.fm_linear(linear_input)  # [batch_size, 1]
        
        # Interaction term (second-order)
        interaction_term = self._fm_interaction(embeddings)
        
        # Deep component
        deep_input = torch.cat([linear_input, conv_pooled], dim=1)
        deep_out = deep_input
        
        for layer in self.deep_layers:
            deep_out = layer(deep_out)
        
        deep_term = self.prediction_layer(deep_out)
        
        # Final prediction
        prediction = linear_term + interaction_term + deep_term
        
        return self.sigmoid(prediction)
    
    def _fm_interaction(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute Factorization Machine interaction term
        
        Args:
            embeddings: List of embedding tensors
            
        Returns:
            Interaction term [batch_size, 1]
        """
        # Sum of embeddings
        sum_emb = torch.stack(embeddings, dim=1).sum(dim=1)  # [batch_size, embedding_dim]
        
        # Sum of squared embeddings
        squared_emb = torch.stack([emb.pow(2) for emb in embeddings], dim=1).sum(dim=1)
        
        # FM interaction: 0.5 * (sum^2 - sum_squares)
        interaction = 0.5 * (sum_emb.pow(2) - squared_emb).sum(dim=1, keepdim=True)
        
        return interaction
    
    def predict_job_scores(self, user_id: int, job_ids: List[int], 
                          skill_ids: List[int], location_id: int, 
                          company_id: int) -> Dict[int, float]:
        """
        Predict job recommendation scores for a user
        
        Args:
            user_id: User ID
            job_ids: List of job IDs to score
            skill_ids: List of user skill IDs
            location_id: Location ID
            company_id: Company ID
            
        Returns:
            Dictionary mapping job_id to score
        """
        self.eval()
        
        # Pad skill_ids to fixed length
        max_skills = 10
        if len(skill_ids) < max_skills:
            skill_ids = skill_ids + [0] * (max_skills - len(skill_ids))
        else:
            skill_ids = skill_ids[:max_skills]
        
        with torch.no_grad():
            scores = {}
            for job_id in job_ids:
                # Create batch tensors
                user_tensor = torch.tensor([user_id], dtype=torch.long)
                job_tensor = torch.tensor([job_id], dtype=torch.long)
                skill_tensor = torch.tensor([skill_ids], dtype=torch.long)
                location_tensor = torch.tensor([location_id], dtype=torch.long)
                company_tensor = torch.tensor([company_id], dtype=torch.long)
                
                # Get prediction
                score = self.forward(user_tensor, job_tensor, skill_tensor, 
                                   location_tensor, company_tensor)
                
                scores[job_id] = score.item()
        
        return scores
    
    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for analysis"""
        return self.user_embedding(user_ids)
    
    def get_job_embeddings(self, job_ids: torch.Tensor) -> torch.Tensor:
        """Get job embeddings for analysis"""
        return self.job_embedding(job_ids)


class ConvFMTrainer:
    """
    Trainer class for ConvFM model
    """
    
    def __init__(self, model: ConvFMJobRecommender, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                batch['user_ids'],
                batch['job_ids'],
                batch['skill_ids'],
                batch['location_ids'],
                batch['company_ids']
            )
            
            # Compute loss
            loss = self.criterion(predictions.squeeze(), batch['labels'].float())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                pred = self.model(
                    batch['user_ids'],
                    batch['job_ids'],
                    batch['skill_ids'],
                    batch['location_ids'],
                    batch['company_ids']
                )
                
                loss = self.criterion(pred.squeeze(), batch['labels'].float())
                total_loss += loss.item()
                
                predictions.extend(pred.cpu().numpy().flatten())
                labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Binary classification metrics
        pred_binary = (predictions > 0.5).astype(int)
        
        accuracy = np.mean(pred_binary == labels)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }


def create_model_from_config(config: Dict) -> ConvFMJobRecommender:
    """
    Create ConvFM model from configuration dictionary
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized ConvFM model
    """
    return ConvFMJobRecommender(
        num_users=config['num_users'],
        num_jobs=config['num_jobs'],
        num_skills=config['num_skills'],
        num_locations=config['num_locations'],
        num_companies=config['num_companies'],
        embedding_dim=config.get('embedding_dim', 64),
        conv_filters=config.get('conv_filters', 64),
        conv_kernel_size=config.get('conv_kernel_size', 3),
        dropout_rate=config.get('dropout_rate', 0.2),
        hidden_dims=config.get('hidden_dims', [128, 64, 32])
    )
