# ConvFM Job Recommendation Model

## Overview

The Convolutional Factorization Machine (ConvFM) is a state-of-the-art recommendation model designed specifically for job recommendations. It combines the strengths of Factorization Machines, Convolutional Neural Networks, and Deep Neural Networks to provide accurate and personalized job recommendations.

## Architecture

### Core Components

1. **Embedding Layers**: Learn dense representations for users, jobs, skills, locations, and companies
2. **Convolutional Component**: Captures sequential patterns in feature interactions
3. **Factorization Machine**: Models second-order feature interactions efficiently
4. **Deep Neural Network**: Learns complex non-linear feature combinations
5. **Prediction Layer**: Combines all components for final recommendation scores

### Mathematical Foundation

The ConvFM model combines three key components:

```
Prediction = Linear Term + FM Interaction Term + Deep Component
```

#### 1. Linear Term
```
y_linear = w₀ + Σ(wᵢ × xᵢ)
```
Where `w₀` is the global bias and `wᵢ` are linear weights.

#### 2. Factorization Machine Interaction Term
```
y_FM = 0.5 × (Σ(vᵢ × xᵢ))² - Σ((vᵢ × xᵢ)²)
```
Where `vᵢ` are embedding vectors for each feature.

#### 3. Deep Component
```
y_deep = DNN(concat(embeddings, conv_features))
```
Where DNN is a multi-layer perceptron with ReLU activations and dropout.

#### 4. Convolutional Component
```
conv_out = Conv1D(embedding_sequence)
pooled = GlobalAveragePooling1D(conv_out)
```

## Model Architecture Diagram

```
Input Features:
├── User ID → User Embedding (64D)
├── Job ID → Job Embedding (64D)
├── Skills → Skill Embeddings (64D × max_skills)
├── Location → Location Embedding (64D)
└── Company → Company Embedding (64D)

Convolutional Branch:
├── Stack embeddings → [batch, 64, 5]
├── Conv1D(64 filters, kernel=3) → [batch, 64, 5]
├── ReLU activation
└── Global Average Pooling → [batch, 64]

Factorization Machine Branch:
├── Linear term: W·x
├── Interaction term: 0.5 × (Σx)² - Σx²
└── Combined: Linear + Interaction

Deep Branch:
├── Concatenate [embeddings, conv_features]
├── Dense(128) → BatchNorm → ReLU → Dropout(0.2)
├── Dense(64) → BatchNorm → ReLU → Dropout(0.2)
├── Dense(32) → BatchNorm → ReLU → Dropout(0.2)
└── Dense(1) → Sigmoid

Final Output:
└── Linear + FM + Deep → Sigmoid → [0,1]
```

## Key Features

### 1. Multi-Field Embeddings
- **User Embeddings**: Capture user preferences and behavior patterns
- **Job Embeddings**: Represent job characteristics and requirements
- **Skill Embeddings**: Model technical skills and competencies
- **Location Embeddings**: Encode geographic preferences
- **Company Embeddings**: Represent company culture and values

### 2. Convolutional Feature Learning
- **1D Convolution**: Captures sequential patterns in feature combinations
- **Multiple Filters**: Learns diverse interaction patterns
- **Global Pooling**: Aggregates spatial information effectively

### 3. Factorization Machine Integration
- **Second-Order Interactions**: Models pairwise feature interactions
- **Parameter Efficiency**: Reduces model complexity compared to full interaction matrix
- **Interpretability**: Provides insights into feature importance

### 4. Deep Neural Network
- **Non-linear Learning**: Captures complex feature combinations
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting
- **Residual Connections**: Helps with gradient flow

## Training Process

### 1. Data Preparation
```python
# Load and preprocess data
jobs_df = pd.read_csv('data/collected_jobs.csv')
interactions_df = pd.read_csv('data/user_interactions.csv')

# Feature extraction
extractor = JobFeatureExtractor()
extractor.fit(jobs_df, interactions_df)
job_features = extractor.transform_jobs(jobs_df)
interaction_features = extractor.transform_interactions(interactions_df)
```

### 2. Model Training
```python
# Initialize model
model = ConvFMJobRecommender(
    num_users=num_users,
    num_jobs=num_jobs,
    num_skills=num_skills,
    num_locations=num_locations,
    num_companies=num_companies,
    embedding_dim=64,
    conv_filters=64,
    hidden_dims=[128, 64, 32]
)

# Training pipeline
trainer = ConvFMTrainer(model, learning_rate=0.001)
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)
```

### 3. Hyperparameter Tuning

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `embedding_dim` | 32-128 | 64 | Embedding dimension |
| `conv_filters` | 32-128 | 64 | Number of conv filters |
| `conv_kernel_size` | 3-7 | 3 | Conv kernel size |
| `dropout_rate` | 0.1-0.5 | 0.2 | Dropout probability |
| `hidden_dims` | [64,32] to [256,128,64] | [128,64,32] | Deep network layers |
| `learning_rate` | 0.0001-0.01 | 0.001 | Adam learning rate |
| `batch_size` | 16-128 | 32 | Training batch size |

## Performance Metrics

### 1. Accuracy Metrics
- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Recall@K**: Proportion of relevant items that are recommended
- **F1-Score@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain

### 2. Business Metrics
- **Click-Through Rate (CTR)**: Percentage of recommendations clicked
- **Conversion Rate**: Percentage of recommendations leading to job applications
- **User Engagement**: Time spent on recommended jobs
- **Diversity**: Variety of recommended job types and companies

### 3. Model Metrics
- **Training Loss**: Binary cross-entropy loss during training
- **Validation Loss**: Loss on held-out validation set
- **AUC-ROC**: Area under the ROC curve
- **Inference Time**: Time to generate recommendations

## Expected Performance

### Accuracy Benchmarks
| Metric | Expected Range | Target |
|--------|----------------|---------|
| Precision@10 | 0.15-0.25 | 0.20 |
| Recall@10 | 0.30-0.50 | 0.40 |
| F1-Score@10 | 0.20-0.35 | 0.28 |
| NDCG@10 | 0.40-0.60 | 0.50 |
| AUC-ROC | 0.75-0.90 | 0.85 |

### Performance Characteristics
- **Training Time**: 10-30 minutes (depending on data size)
- **Memory Usage**: 2-4 GB RAM
- **Model Size**: 10-50 MB (depending on vocabulary size)
- **Inference Speed**: <10ms per prediction
- **Throughput**: 100-1000 predictions/second

## Usage Examples

### 1. Basic Training
```python
from ml_models.training_pipeline import ConvFMTrainingPipeline, create_training_config

# Create configuration
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
```

### 2. Making Predictions
```python
# Load trained model
model = ConvFMJobRecommender(...)
model.load_state_dict(torch.load('models/best_convfm_model.pt'))

# Generate recommendations
user_id = 123
job_ids = [1, 2, 3, 4, 5]
user_skills = [10, 20, 30]
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
top_jobs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
```

### 3. API Usage
```python
import requests

# Get recommendations via API
response = requests.post('http://localhost:8001/recommend', json={
    'user_id': 123,
    'max_recommendations': 10,
    'location_preference': 'San Francisco',
    'salary_min': 100000
})

recommendations = response.json()['recommendations']
```

## Advanced Features

### 1. Cold Start Handling
- **New Users**: Use demographic features and preferences
- **New Jobs**: Leverage job content and company information
- **Popularity Bias**: Include trending and popular jobs

### 2. Real-time Learning
- **Online Updates**: Incremental model updates with new data
- **A/B Testing**: Compare different model versions
- **Feedback Loop**: Incorporate user feedback into recommendations

### 3. Explainability
- **Feature Importance**: Identify which features drive recommendations
- **Attention Weights**: Visualize model focus areas
- **Counterfactual Analysis**: Understand what-if scenarios

## Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Increase embedding dimensions
   - Add more training data
   - Tune hyperparameters
   - Check data quality

2. **Overfitting**
   - Increase dropout rate
   - Reduce model complexity
   - Add regularization
   - Use early stopping

3. **Slow Training**
   - Reduce batch size
   - Use GPU acceleration
   - Optimize data loading
   - Reduce model size

4. **Memory Issues**
   - Reduce embedding dimensions
   - Use gradient checkpointing
   - Implement data streaming
   - Optimize batch size

### Debugging Tips

1. **Monitor Training Metrics**
   - Watch for loss plateaus
   - Check validation performance
   - Analyze learning curves

2. **Data Validation**
   - Verify feature distributions
   - Check for data leakage
   - Validate preprocessing steps

3. **Model Inspection**
   - Visualize embeddings
   - Analyze attention patterns
   - Test with synthetic data

## Future Improvements

### 1. Architecture Enhancements
- **Graph Neural Networks**: Model user-job relationships as graphs
- **Transformer Architecture**: Use self-attention for sequence modeling
- **Multi-task Learning**: Joint optimization of multiple objectives

### 2. Feature Engineering
- **Temporal Features**: Incorporate time-based patterns
- **Contextual Features**: Add session and context information
- **External Data**: Integrate market trends and economic indicators

### 3. Optimization Techniques
- **Neural Architecture Search**: Automatically find optimal architectures
- **Knowledge Distillation**: Compress models for deployment
- **Federated Learning**: Train on distributed data sources

## References

1. Rendle, S. (2010). Factorization machines. ICDM 2010.
2. He, X., & Chua, T. S. (2017). Neural factorization machines for sparse predictive analytics. SIGIR 2017.
3. Zhang, W., et al. (2019). Deep learning over multi-field categorical data. ECIR 2016.
4. Guo, H., et al. (2017). DeepFM: A factorization-machine based neural network for CTR prediction. IJCAI 2017.
