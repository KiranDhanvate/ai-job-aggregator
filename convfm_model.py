"""
ConvFM Model: Hybrid CNN + Factorization Machine for Job Recommendation
Implements the architecture shown in the workflow diagram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class TextCNN(nn.Module):
    """
    CNN for extracting features from job descriptions and user profiles
    Implements Phase 2 from the workflow diagram
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 128,
        filter_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super(TextCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # 2. Convolutional Layers (multiple filter sizes for n-gram features)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        # 3. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension after concatenating all conv outputs
        self.output_dim = num_filters * len(filter_sizes)
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: [batch_size, seq_length]
        Returns:
            features: [batch_size, output_dim]
        """
        # Embedding: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(text)
        
        # Add channel dimension: [batch_size, 1, seq_length, embedding_dim]
        embedded = embedded.unsqueeze(1)
        
        # Apply convolution + ReLU + max pooling for each filter size
        conv_outputs = []
        for conv in self.convs:
            # Conv: [batch_size, num_filters, seq_length - filter_size + 1, 1]
            conv_out = F.relu(conv(embedded))
            
            # Max pooling: [batch_size, num_filters, 1, 1]
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))
            
            # Squeeze: [batch_size, num_filters]
            conv_outputs.append(pooled.squeeze(3).squeeze(2))
        
        # Concatenate all conv outputs: [batch_size, num_filters * num_filter_sizes]
        features = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        features = self.dropout(features)
        
        return features


class FactorizationMachine(nn.Module):
    """
    Factorization Machine for modeling feature interactions
    Implements Phase 3 from the workflow diagram
    """
    
    def __init__(self, input_dim: int, factor_dim: int = 64):
        super(FactorizationMachine, self).__init__()
        
        # Linear weights for first-order interactions
        self.linear = nn.Linear(input_dim, 1, bias=True)
        
        # Embedding for second-order interactions
        # Each feature gets a factor vector
        self.factor_embeddings = nn.Parameter(
            torch.randn(input_dim, factor_dim)
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.factor_embeddings)
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] - concatenated user and job features
        Returns:
            prediction: [batch_size, 1] - relevance score
        """
        batch_size = x.size(0)
        
        # First-order: w0 + sum(wi * xi)
        linear_terms = self.linear(x)  # [batch_size, 1]
        
        # Second-order: 0.5 * sum_f [(sum_i vi,f * xi)^2 - sum_i (vi,f * xi)^2]
        # This captures all pairwise interactions efficiently
        
        # Expand x for broadcasting: [batch_size, input_dim, 1]
        x_expanded = x.unsqueeze(2)
        
        # Factor embeddings: [input_dim, factor_dim]
        # Multiply features with factor vectors: [batch_size, input_dim, factor_dim]
        factor_mul = x_expanded * self.factor_embeddings.unsqueeze(0)
        
        # Square of sum
        sum_square = torch.sum(factor_mul, dim=1) ** 2  # [batch_size, factor_dim]
        
        # Sum of squares
        square_sum = torch.sum(factor_mul ** 2, dim=1)  # [batch_size, factor_dim]
        
        # Interaction terms
        interaction_terms = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)
        
        # Final prediction
        output = linear_terms + interaction_terms  # [batch_size, 1]
        
        return output


class ConvFM(nn.Module):
    """
    Complete ConvFM Model: CNN for feature extraction + FM for recommendation
    Integrates all phases from the workflow diagram
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        cnn_num_filters: int = 128,
        cnn_filter_sizes: List[int] = [3, 4, 5],
        fm_factor_dim: int = 64,
        additional_features_dim: int = 50,  # For categorical features (location, job type, etc.)
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        super(ConvFM, self).__init__()
        
        # Phase 2: CNN for text feature extraction
        self.text_cnn = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=cnn_num_filters,
            filter_sizes=cnn_filter_sizes,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        # Embedding layers for categorical features
        self.additional_embeddings = nn.ModuleDict({
            'job_type': nn.Embedding(20, 16),  # Assuming 20 job types
            'location': nn.Embedding(500, 32),  # Assuming 500 locations
            'experience_level': nn.Embedding(10, 8),  # 10 experience levels
            'education_level': nn.Embedding(10, 8),  # 10 education levels
            'industry': nn.Embedding(100, 16),  # 100 industries
        })
        
        # Calculate total feature dimension
        self.cnn_output_dim = self.text_cnn.output_dim
        self.categorical_dim = 16 + 32 + 8 + 8 + 16  # Sum of embedding dims
        
        # Total dimension = User CNN features + Job CNN features + Categorical features
        self.total_feature_dim = (self.cnn_output_dim * 2) + (self.categorical_dim * 2) + additional_features_dim
        
        # Phase 3: Factorization Machine for recommendation
        self.fm = FactorizationMachine(
            input_dim=self.total_feature_dim,
            factor_dim=fm_factor_dim
        )
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(self.total_feature_dim)
        
    def extract_cnn_features(
        self,
        text: torch.Tensor,
        categorical_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract features from text and categorical data
        
        Args:
            text: [batch_size, seq_length]
            categorical_features: Dict of categorical feature tensors
        Returns:
            combined_features: [batch_size, cnn_output_dim + categorical_dim]
        """
        # Extract CNN features from text
        text_features = self.text_cnn(text)  # [batch_size, cnn_output_dim]
        
        # Extract categorical features
        cat_features = []
        for name, embedding_layer in self.additional_embeddings.items():
            if name in categorical_features:
                embedded = embedding_layer(categorical_features[name])
                cat_features.append(embedded)
        
        # Concatenate all categorical features
        if cat_features:
            cat_features = torch.cat(cat_features, dim=1)
            combined = torch.cat([text_features, cat_features], dim=1)
        else:
            combined = text_features
        
        return combined
    
    def forward(
        self,
        user_text: torch.Tensor,
        job_text: torch.Tensor,
        user_categorical: Dict[str, torch.Tensor],
        job_categorical: Dict[str, torch.Tensor],
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Complete forward pass through ConvFM
        
        Args:
            user_text: User profile/resume text [batch_size, seq_length]
            job_text: Job description text [batch_size, seq_length]
            user_categorical: Dict of user categorical features
            job_categorical: Dict of job categorical features
            additional_features: Additional numeric features [batch_size, additional_features_dim]
        
        Returns:
            predictions: [batch_size, 1] - Job relevance scores
        """
        # Extract features for user profile
        user_features = self.extract_cnn_features(user_text, user_categorical)
        
        # Extract features for job description
        job_features = self.extract_cnn_features(job_text, job_categorical)
        
        # Concatenate user and job features
        combined_features = torch.cat([user_features, job_features], dim=1)
        
        # Add additional numeric features if provided
        if additional_features is not None:
            combined_features = torch.cat([combined_features, additional_features], dim=1)
        
        # Apply batch normalization
        combined_features = self.batch_norm(combined_features)
        
        # Pass through Factorization Machine
        predictions = self.fm(combined_features)
        
        return predictions
    
    def predict_top_k(
        self,
        user_text: torch.Tensor,
        job_texts: torch.Tensor,
        user_categorical: Dict[str, torch.Tensor],
        job_categoricals: List[Dict[str, torch.Tensor]],
        additional_features: Optional[torch.Tensor] = None,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k job recommendations for a user
        
        Returns:
            top_k_scores: [k] - Top k relevance scores
            top_k_indices: [k] - Indices of top k jobs
        """
        self.eval()
        with torch.no_grad():
            # Repeat user features for all jobs
            batch_size = job_texts.size(0)
            user_text_repeated = user_text.repeat(batch_size, 1)
            
            # Prepare user categorical features
            user_cat_repeated = {}
            for key, value in user_categorical.items():
                user_cat_repeated[key] = value.repeat(batch_size)
            
            # Compute predictions for all jobs
            predictions = []
            for i in range(batch_size):
                job_cat = {k: v[i].unsqueeze(0) for k, v in job_categoricals[i].items()}
                
                pred = self.forward(
                    user_text_repeated[i:i+1],
                    job_texts[i:i+1],
                    user_categorical,
                    job_cat,
                    additional_features[i:i+1] if additional_features is not None else None
                )
                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0).squeeze()
            
            # Get top-k
            top_k_scores, top_k_indices = torch.topk(predictions, k=min(k, len(predictions)))
        
        return top_k_scores, top_k_indices


def create_convfm_model(
    vocab_size: int,
    pretrained_embeddings: Optional[np.ndarray] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ConvFM:
    """
    Factory function to create ConvFM model with recommended hyperparameters
    """
    model = ConvFM(
        vocab_size=vocab_size,
        embedding_dim=300,
        cnn_num_filters=128,
        cnn_filter_sizes=[3, 4, 5],
        fm_factor_dim=64,
        additional_features_dim=50,
        dropout=0.5,
        pretrained_embeddings=pretrained_embeddings
    )
    
    model = model.to(device)
    return model


# Loss function for training
class ConvFMLoss(nn.Module):
    """
    Combined loss for ConvFM: MSE for rating prediction + regularization
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        super(ConvFMLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: ConvFM
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, 1]
            targets: [batch_size, 1]
            model: ConvFM model for regularization
        """
        # Primary loss: Mean Squared Error
        mse = self.mse_loss(predictions, targets)
        
        # L2 regularization on FM factor embeddings
        fm_reg = torch.norm(model.fm.factor_embeddings) ** 2
        
        # Total loss
        total_loss = mse + self.lambda_reg * fm_reg
        
        return total_loss