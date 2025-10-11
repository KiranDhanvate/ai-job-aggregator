# ML Models Package

This package contains machine learning models and utilities for the AI Job Aggregator recommendation system.

## Overview

The ML models package implements a Convolutional Factorization Machine (ConvFM) for job recommendation, combining the strengths of:

- **Factorization Machines**: For modeling feature interactions
- **Convolutional Neural Networks**: For sequence modeling and pattern recognition
- **Deep Neural Networks**: For non-linear feature learning

## Components

### 1. ConvFM Model (`convfm_model.py`)

The core recommendation model that combines:
- User embeddings
- Job embeddings  
- Skill embeddings
- Location embeddings
- Company embeddings

**Key Features:**
- Multi-field embeddings for different entity types
- Convolutional layers for sequence modeling
- Factorization Machine for second-order interactions
- Deep neural network for complex feature learning
- Dropout and batch normalization for regularization

### 2. Feature Extractor (`feature_extractor.py`)

Handles data preprocessing and feature engineering:

**Features Extracted:**
- User features (interaction history, preferences)
- Job features (skills, location, company, salary)
- Skill features (TF-IDF, embeddings)
- Text features (job descriptions)
- Categorical features (job type, experience level)

**Preprocessing:**
- Label encoding for categorical variables
- Text vectorization using TF-IDF
- Skill extraction from job descriptions
- Feature scaling and normalization

### 3. Training Pipeline (`training_pipeline.py`)

Complete training workflow including:

**Training Process:**
- Data preparation and splitting
- Model initialization
- Training with early stopping
- Validation and evaluation
- Model saving and loading

**Evaluation Metrics:**
- Binary classification accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Loss tracking

## Usage

### Basic Training

```python
from ml_models.training_pipeline import ConvFMTrainingPipeline, create_training_config
import pandas as pd

# Load your data
jobs_df = pd.read_csv('data/collected_jobs.csv')
interactions_df = pd.read_csv('data/user_interactions.csv')

# Create training configuration
config = create_training_config(
    embedding_dim=64,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001
)

# Initialize pipeline
pipeline = ConvFMTrainingPipeline(config)

# Run training
results = pipeline.run_full_pipeline(jobs_df, interactions_df)

print(f"Model saved to: {results['model_path']}")
print(f"Validation accuracy: {results['validation_metrics']['accuracy']:.4f}")
```

### Making Recommendations

```python
from ml_models.convfm_model import ConvFMJobRecommender
import torch

# Load trained model
model = ConvFMJobRecommender(...)
model.load_state_dict(torch.load('models/best_convfm_model.pt'))

# Generate recommendations
user_id = 123
job_ids = [1, 2, 3, 4, 5]
user_skills = [10, 20, 30]  # Skill IDs
location_id = 1
company_id = 5

scores = model.predict_job_scores(
    user_id=user_id,
    job_ids=job_ids,
    skill_ids=user_skills,
    location_id=location_id,
    company_id=company_id
)

# Get top recommendations
top_jobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top recommendations: {top_jobs}")
```

### Feature Extraction

```python
from ml_models.feature_extractor import JobFeatureExtractor

# Initialize extractor
extractor = JobFeatureExtractor(max_skills=50)

# Fit on training data
extractor.fit(jobs_df, interactions_df)

# Transform new data
job_features = extractor.transform_jobs(new_jobs_df)
interaction_features = extractor.transform_interactions(new_interactions_df)

# Save for later use
extractor.save('artifacts/feature_extractor.pkl')
```

## Model Architecture

### ConvFM Architecture

```
Input Features:
├── User ID → User Embedding (64D)
├── Job ID → Job Embedding (64D)  
├── Skills → Skill Embeddings (64D × max_skills)
├── Location → Location Embedding (64D)
└── Company → Company Embedding (64D)

Convolutional Component:
├── Stack embeddings → [batch, 64, 5]
├── Conv1D(filters=64, kernel=3) → [batch, 64, 5]
├── ReLU activation
└── Adaptive pooling → [batch, 64]

Factorization Machine:
├── Linear term: W·x
├── Interaction term: 0.5 × (Σx)² - Σx²
└── Combined: Linear + Interaction

Deep Component:
├── Concatenate [embeddings, conv_features]
├── Dense(128) → BatchNorm → ReLU → Dropout
├── Dense(64) → BatchNorm → ReLU → Dropout  
├── Dense(32) → BatchNorm → ReLU → Dropout
└── Dense(1) → Sigmoid

Final Output:
└── Linear + FM + Deep → Sigmoid → [0,1]
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Embedding dimension for all entities |
| `conv_filters` | 64 | Number of convolutional filters |
| `conv_kernel_size` | 3 | Convolutional kernel size |
| `dropout_rate` | 0.2 | Dropout rate for regularization |
| `hidden_dims` | [128, 64, 32] | Deep network layer sizes |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 50 | Maximum training epochs |
| `patience` | 10 | Early stopping patience |

## Performance

### Expected Performance

- **Training Time**: 10-30 minutes (depending on data size)
- **Memory Usage**: 2-4 GB RAM
- **Model Size**: 10-50 MB (depending on vocabulary size)
- **Inference Speed**: <1ms per prediction

### Accuracy Benchmarks

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 0.75-0.85 |
| Precision | 0.70-0.80 |
| Recall | 0.65-0.75 |
| F1-Score | 0.67-0.77 |
| ROC-AUC | 0.80-0.90 |

## File Structure

```
ml_models/
├── __init__.py              # Package initialization
├── convfm_model.py          # ConvFM model implementation
├── feature_extractor.py     # Feature extraction utilities
├── training_pipeline.py     # Training pipeline
└── README.md               # This documentation
```

## Dependencies

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0

## Future Improvements

- [ ] Multi-task learning for different recommendation scenarios
- [ ] Graph neural networks for user-job relationship modeling
- [ ] Transformer-based architecture for sequence modeling
- [ ] Online learning for real-time model updates
- [ ] Explainable AI for recommendation interpretability
- [ ] A/B testing framework for model evaluation
