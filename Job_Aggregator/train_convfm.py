"""
Main Training Script for ConvFM Model
Run this script to train the recommendation model
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from convfm_model import create_convfm_model
from feature_extractor import FeatureExtractor
from training_pipeline import (
    ConvFMTrainer,
    prepare_training_data,
    create_sample_user_profiles
)


def main(args):
    """Main training function"""
    
    print("=" * 80)
    print("ConvFM Training Pipeline - AI Job Recommendation System")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    model_dir = Path(args.model_dir)
    artifacts_dir = Path(args.artifacts_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    print("\n📊 Initializing Feature Extractor...")
    feature_extractor = FeatureExtractor()
    
    # Prepare data
    print("\n📦 Preparing Training Data...")
    if args.jobs_csv:
        # Load from provided CSV
        train_loader, val_loader, test_loader = prepare_training_data(
            jobs_csv_path=args.jobs_csv,
            feature_extractor=feature_extractor,
            val_split=args.val_split,
            test_split=args.test_split
        )
    else:
        print("⚠️  No jobs CSV provided. Using sample data...")
        # For demonstration, scrape some jobs
        from jobspy import scrape_jobs
        
        print("  Scraping sample jobs...")
        jobs_df = scrape_jobs(
            site_name=['indeed', 'linkedin'],
            search_term='software engineer',
            location='United States',
            results_wanted=50,
            linkedin_fetch_description=True,
            description_format='plain',
            verbose=0
        )
        
        # Save scraped jobs
        jobs_csv_path = artifacts_dir / 'scraped_jobs.csv'
        jobs_df.to_csv(jobs_csv_path, index=False)
        print(f"  ✅ Saved {len(jobs_df)} jobs to {jobs_csv_path}")
        
        # Prepare data
        train_loader, val_loader, test_loader = prepare_training_data(
            jobs_csv_path=str(jobs_csv_path),
            feature_extractor=feature_extractor,
            val_split=args.val_split,
            test_split=args.test_split
        )
    
    # Save feature extractor
    print("\n💾 Saving Feature Extractor...")
    feature_extractor.save(str(artifacts_dir / 'feature_extractor.pkl'))
    
    # Create model
    print("\n🏗️  Creating ConvFM Model...")
    vocab_size = len(feature_extractor.text_preprocessor.vocab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_convfm_model(vocab_size=vocab_size, device=device)
    
    # Print model architecture
    print("\n📐 Model Architecture:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  CNN Output Dim: {model.cnn_output_dim}")
    print(f"  Total Feature Dim: {model.total_feature_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    print("\n🎯 Initializing Trainer...")
    trainer = ConvFMTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train model
    print("\n🚀 Starting Training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_dir=str(model_dir)
    )
    
    # Plot training history
    print("\n📊 Plotting Training History...")
    trainer.plot_training_history(
        save_path=str(model_dir / 'training_curves.png')
    )
    
    # Evaluate on test set
    print("\n🧪 Evaluating on Test Set...")
    test_loss, test_rmse, test_mae = trainer.validate(test_loader)
    
    print(f"\n{'=' * 80}")
    print("📈 FINAL TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"{'=' * 80}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'batch_size': 32
        }
    }
    
    import json
    with open(model_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Training Complete!")
    print(f"📁 Model saved to: {model_dir}")
    print(f"📁 Artifacts saved to: {artifacts_dir}")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ConvFM Job Recommendation Model')
    
    # Data parameters
    parser.add_argument('--jobs-csv', type=str, default=None,
                        help='Path to CSV file with job postings')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation set proportion')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test set proportion')
    
    # Model parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Output parameters
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts',
                        help='Directory to save artifacts (feature extractor, etc.)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(args)