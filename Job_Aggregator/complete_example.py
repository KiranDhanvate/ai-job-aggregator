"""
Complete ConvFM Workflow Example
This script demonstrates the entire pipeline from data collection to recommendations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def step1_data_collection():
    """
    Step 1: Collect job data from multiple sources
    """
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)
    
    from jobspy import scrape_jobs
    
    print("\n📥 Scraping jobs from multiple platforms...")
    
    # Scrape jobs from different search terms to get diverse data
    search_terms = [
        'python developer',
        'machine learning engineer',
        'data scientist',
        'backend engineer',
        'frontend developer'
    ]
    
    all_jobs = []
    
    for term in search_terms:
        print(f"\n  Searching for: {term}")
        try:
            jobs_df = scrape_jobs(
                site_name=['linkedin', 'indeed'],
                search_term=term,
                location='United States',
                results_wanted=20,
                is_remote=True,
                hours_old=168,  # Last week
                linkedin_fetch_description=True,
                description_format='plain',
                verbose=0
            )
            
            print(f"    ✅ Found {len(jobs_df)} jobs")
            all_jobs.append(jobs_df)
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Combine all jobs
    if all_jobs:
        combined_jobs = pd.concat(all_jobs, ignore_index=True)
        
        # Remove duplicates based on job URL
        combined_jobs = combined_jobs.drop_duplicates(subset=['job_url'], keep='first')
        
        # Save to CSV
        output_path = Path('./data/collected_jobs.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_jobs.to_csv(output_path, index=False)
        
        print(f"\n✅ Total unique jobs collected: {len(combined_jobs)}")
        print(f"📁 Saved to: {output_path}")
        
        return str(output_path)
    else:
        print("\n❌ No jobs collected")
        return None


def step2_train_model(jobs_csv_path):
    """
    Step 2: Train the ConvFM recommendation model
    """
    print("\n" + "="*80)
    print("STEP 2: TRAIN CONVFM MODEL")
    print("="*80)
    
    from convfm_model import create_convfm_model
    from feature_extractor import FeatureExtractor
    from training_pipeline import ConvFMTrainer, prepare_training_data
    
    print("\n🔧 Initializing feature extractor...")
    feature_extractor = FeatureExtractor()
    
    print("\n📦 Preparing training data...")
    train_loader, val_loader, test_loader = prepare_training_data(
        jobs_csv_path=jobs_csv_path,
        feature_extractor=feature_extractor,
        val_split=0.2,
        test_split=0.1
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Save feature extractor
    artifacts_dir = Path('./artifacts')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor.save(str(artifacts_dir / 'feature_extractor.pkl'))
    
    print("\n🏗️  Creating ConvFM model...")
    vocab_size = len(feature_extractor.text_preprocessor.vocab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_convfm_model(vocab_size=vocab_size, device=device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    
    print("\n🚀 Training model...")
    trainer = ConvFMTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.01
    )
    
    # Train with fewer epochs for demo
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,  # Increase to 50+ for production
        early_stopping_patience=7,
        save_dir='./models'
    )
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_loss, test_rmse, test_mae = trainer.validate(test_loader)
    
    print(f"\n📊 Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    
    # Plot training curves
    trainer.plot_training_history(save_path='./models/training_curves.png')
    
    print("\n✅ Model training complete!")


def step3_make_recommendations():
    """
    Step 3: Make personalized job recommendations
    """
    print("\n" + "="*80)
    print("STEP 3: GENERATE RECOMMENDATIONS")
    print("="*80)
    
    from convfm_model import create_convfm_model
    from feature_extractor import FeatureExtractor
    
    # Load feature extractor
    print("\n📥 Loading feature extractor...")
    feature_extractor = FeatureExtractor()
    feature_extractor.load('./artifacts/feature_extractor.pkl')
    
    # Load model
    print("📥 Loading trained model...")
    vocab_size = len(feature_extractor.text_preprocessor.vocab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_convfm_model(vocab_size=vocab_size, device=device)
    
    checkpoint = torch.load('./models/best_convfm_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Load jobs
    print("\n📂 Loading job data...")
    jobs_df = pd.read_csv('./data/collected_jobs.csv')
    jobs_list = jobs_df.to_dict(orient='records')
    print(f"  Loaded {len(jobs_list)} jobs")
    
    # Define multiple user profiles for demonstration
    user_profiles = [
        {
            'name': 'Sarah - ML Engineer',
            'profile': {
                'resume': 'Experienced ML engineer with deep learning expertise',
                'skills': ['python', 'tensorflow', 'pytorch', 'aws', 'kubernetes'],
                'experience_years': 5,
                'education_level': 'masters',
                'preferred_location': 'Remote',
                'open_to_remote': True,
                'expected_min_salary': 130000,
                'expected_max_salary': 180000,
                'preferred_job_type': 'fulltime'
            }
        },
        {
            'name': 'Mike - Backend Developer',
            'profile': {
                'resume': 'Backend developer specializing in microservices',
                'skills': ['java', 'spring', 'postgresql', 'docker', 'redis'],
                'experience_years': 3,
                'education_level': 'bachelors',
                'preferred_location': 'San Francisco',
                'open_to_remote': True,
                'expected_min_salary': 100000,
                'expected_max_salary': 140000,
                'preferred_job_type': 'fulltime'
            }
        },
        {
            'name': 'Emily - Data Scientist',
            'profile': {
                'resume': 'Data scientist with strong analytics background',
                'skills': ['python', 'sql', 'tableau', 'scikit-learn', 'pandas'],
                'experience_years': 2,
                'education_level': 'masters',
                'preferred_location': 'New York',
                'open_to_remote': True,
                'expected_min_salary': 90000,
                'expected_max_salary': 130000,
                'preferred_job_type': 'fulltime'
            }
        }
    ]
    
    # Generate recommendations for each user
    for user_data in user_profiles:
        print(f"\n{'='*80}")
        print(f"Recommendations for: {user_data['name']}")
        print(f"{'='*80}")
        
        user_profile = user_data['profile']
        
        # Extract features
        user_features = feature_extractor.extract_user_features(user_profile)
        job_features_list = [
            feature_extractor.extract_job_features(job) 
            for job in jobs_list
        ]
        
        # Prepare batch data
        batch_data = feature_extractor.prepare_batch_data(
            user_features,
            job_features_list,
            device=device
        )
        
        # Get predictions
        with torch.no_grad():
            predictions = []
            for i in range(len(jobs_list)):
                pred = model(
                    batch_data['user_text'][i:i+1],
                    batch_data['job_text'][i:i+1],
                    {k: v[i:i+1] for k, v in batch_data['user_categorical'].items()},
                    [batch_data['job_categoricals'][i]],
                    batch_data['additional_features'][i:i+1]
                )
                predictions.append(pred.item())
        
        # Get top 5 recommendations
        top_5_indices = np.argsort(predictions)[::-1][:5]
        
        print(f"\nUser Skills: {', '.join(user_profile['skills'])}")
        print(f"Experience: {user_profile['experience_years']} years")
        print(f"Salary Range: ${user_profile['expected_min_salary']:,} - ${user_profile['expected_max_salary']:,}")
        
        print(f"\n🎯 Top 5 Job Recommendations:\n")
        
        for rank, idx in enumerate(top_5_indices, 1):
            job = jobs_list[idx]
            match_score = predictions[idx]
            
            # Calculate skill match
            user_skills = set(s.lower() for s in user_profile['skills'])
            job_skills = set(s.lower() for s in job_features_list[idx].get('skills', []))
            matching_skills = list(user_skills.intersection(job_skills))
            
            print(f"{rank}. {job['title']}")
            print(f"   Company: {job['company']}")
            print(f"   Location: {job['location']}")
            print(f"   Match Score: {match_score:.4f}")
            print(f"   Matching Skills ({len(matching_skills)}): {', '.join(matching_skills[:5])}")
            
            if job.get('min_amount') and job.get('max_amount'):
                salary_range = f"${int(job['min_amount']):,} - ${int(job['max_amount']):,}"
                print(f"   Salary: {salary_range}")
            
            print(f"   URL: {job['job_url']}")
            print()


def step4_api_demo():
    """
    Step 4: Demonstrate API usage
    """
    print("\n" + "="*80)
    print("STEP 4: API DEMONSTRATION")
    print("="*80)
    
    print("\n📡 Starting API server...")
    print("To test the API, run in a separate terminal:")
    print("\n  python recommendation_api.py\n")
    
    print("Then use these curl commands:\n")
    
    # Example 1: Health check
    print("1️⃣  Health Check:")
    print("   curl http://localhost:5000/health\n")
    
    # Example 2: Scrape jobs
    print("2️⃣  Scrape Jobs:")
    print('''   curl -X POST http://localhost:5000/scrape \\
     -H "Content-Type: application/json" \\
     -d '{
       "search_term": "python developer",
       "location": "Remote",
       "results_wanted": 10
     }'
   ''')
    
    # Example 3: Get recommendations
    print("\n3️⃣  Get Recommendations:")
    print('''   curl -X POST http://localhost:5000/scrape-and-recommend \\
     -H "Content-Type: application/json" \\
     -d '{
       "search_term": "data scientist",
       "location": "Remote",
       "results_wanted": 20,
       "user_profile": {
         "skills": ["python", "machine learning", "sql"],
         "experience_years": 3,
         "education_level": "masters",
         "expected_min_salary": 100000,
         "expected_max_salary": 150000
       },
       "top_k": 10
     }'
   ''')
    
    print("\n💡 Or visit http://localhost:5000/docs for interactive API documentation")


def main():
    """
    Main function to run complete workflow
    """
    print("\n" + "🎯"*40)
    print("ConvFM Job Recommendation System - Complete Workflow")
    print("🎯"*40)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create necessary directories
    Path('./data').mkdir(exist_ok=True)
    Path('./models').mkdir(exist_ok=True)
    Path('./artifacts').mkdir(exist_ok=True)
    
    try:
        # Step 1: Collect data
        jobs_csv_path = step1_data_collection()
        
        if jobs_csv_path is None:
            print("\n❌ Data collection failed. Using sample data...")
            # Check if we have existing data
            if Path('./data/collected_jobs.csv').exists():
                jobs_csv_path = './data/collected_jobs.csv'
                print("✅ Found existing job data")
            else:
                print("❌ No job data available. Exiting...")
                return
        
        # Step 2: Train model
        print("\n⏳ Proceeding to model training...")
        input("Press Enter to continue (or Ctrl+C to skip)...")
        step2_train_model(jobs_csv_path)
        
        # Step 3: Make recommendations
        print("\n⏳ Proceeding to generate recommendations...")
        input("Press Enter to continue (or Ctrl+C to skip)...")
        step3_make_recommendations()
        
        # Step 4: API demo
        print("\n⏳ Proceeding to API demonstration...")
        input("Press Enter to see API examples (or Ctrl+C to skip)...")
        step4_api_demo()
        
        print("\n" + "="*80)
        print("✅ WORKFLOW COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the generated recommendations above")
        print("2. Check training curves: ./models/training_curves.png")
        print("3. Start the API server: python recommendation_api.py")
        print("4. Integrate with your frontend application")
        print("\n📚 For more details, see the documentation in README.md")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()