"""
Enhanced API Server with ConvFM-based Job Recommendations
Integrates with the existing Flask/FastAPI server
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from jobspy import scrape_jobs
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

# Import our ConvFM components
from convfm_model import ConvFM, create_convfm_model
from feature_extractor import FeatureExtractor

app = Flask(__name__)
CORS(app)

# Global variables for model and feature extractor
model = None
feature_extractor = None
model_loaded = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration
MODEL_DIR = Path('./models')
ARTIFACTS_DIR = Path('./artifacts')


def load_model_and_extractor():
    """Load trained ConvFM model and feature extractor"""
    global model, feature_extractor, model_loaded
    
    try:
        print("Loading ConvFM model and feature extractor...")
        
        # Load feature extractor
        feature_extractor = FeatureExtractor()
        extractor_path = ARTIFACTS_DIR / 'feature_extractor.pkl'
        
        if extractor_path.exists():
            feature_extractor.load(str(extractor_path))
            
            # Load model
            vocab_size = len(feature_extractor.text_preprocessor.vocab)
            model = create_convfm_model(vocab_size, device=device)
            
            # Load model weights
            model_path = MODEL_DIR / 'best_convfm_model.pt'
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model_loaded = True
                print(f"✅ Model loaded successfully from {model_path}")
            else:
                print("⚠️  Model checkpoint not found. Using untrained model.")
        else:
            print("⚠️  Feature extractor not found. Model not loaded.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False


def clean_description(html_text):
    """Remove HTML tags and clean text"""
    if not html_text or pd.isna(html_text):
        return ""
    from bs4 import BeautifulSoup
    import re
    
    soup = BeautifulSoup(str(html_text), 'html.parser')
    text = soup.get_text(separator=' ')
    text = text.replace('&nbsp;', ' ').replace('&rsquo;', "'")
    text = text.replace('&ldquo;', '"').replace('&rdquo;', '"')
    text = text.replace('&ndash;', '-').replace('&mdash;', '—')
    text = text.replace('&#8209;', '-')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "AI Job Aggregator & Recommender API with ConvFM",
        "version": "3.1.0",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "endpoints": {
            "/health": "GET - Check API health",
            "/scrape": "POST - Scrape jobs from job boards",
            "/recommend": "POST - Get AI-powered job recommendations (ConvFM)",
            "/scrape-and-recommend": "POST - Scrape jobs and get recommendations",
            "/stats": "GET - Get API statistics",
            "/model/info": "GET - Get model information"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Job Aggregator & Recommender",
        "model_loaded": model_loaded,
        "device": device,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "Please train the model first using the training pipeline"
        }), 503
    
    return jsonify({
        "model_type": "ConvFM (Hybrid CNN + Factorization Machine)",
        "status": "loaded",
        "vocabulary_size": len(feature_extractor.text_preprocessor.vocab),
        "device": device,
        "components": {
            "cnn": "Text feature extraction from job descriptions and resumes",
            "fm": "Factorization Machine for interaction modeling"
        }
    })


@app.route('/scrape', methods=['POST', 'GET'])
def scrape():
    """Scrape jobs from job boards"""
    try:
        if request.method == 'POST':
            data = request.get_json()
        else:
            data = request.args.to_dict()
            # Convert string arrays
            if 'site_name' in data:
                data['site_name'] = data['site_name'].split(',')
        
        # Extract parameters
        site_name = data.get('site_name', ['indeed', 'linkedin'])
        search_term = data.get('search_term', 'python developer')
        location = data.get('location', 'United States')
        results_wanted = int(data.get('results_wanted', 10))
        is_remote = data.get('is_remote', False)
        job_type = data.get('job_type', None)
        hours_old = data.get('hours_old', None)
        country_indeed = data.get('country_indeed', 'usa')
        description_format = data.get('description_format', 'plain')
        
        # Scrape jobs
        jobs_df = scrape_jobs(
            site_name=site_name,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            is_remote=is_remote,
            job_type=job_type,
            hours_old=hours_old,
            country_indeed=country_indeed,
            linkedin_fetch_description=True,
            description_format=description_format,
            verbose=0
        )
        
        # Convert to list
        jobs_list = jobs_df.to_dict(orient='records')
        
        # Clean descriptions
        for job in jobs_list:
            if 'description' in job and description_format == 'plain':
                job['description_clean'] = clean_description(job['description'])
        
        return jsonify({
            "success": True,
            "count": len(jobs_list),
            "jobs": jobs_list,
            "search_params": {
                "search_term": search_term,
                "location": location,
                "sites": site_name
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Get AI-powered job recommendations using ConvFM
    
    Request body:
    {
        "user_profile": {
            "resume": "text...",
            "skills": ["python", "ml", ...],
            "experience_years": 3,
            "education_level": "bachelors",
            "preferred_location": "Remote",
            "expected_salary_min": 80000,
            "expected_salary_max": 120000,
            ...
        },
        "jobs": [...],  // List of job postings
        "top_k": 10  // Number of recommendations
    }
    """
    if not model_loaded:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please train the model first."
        }), 503
    
    try:
        data = request.get_json()
        
        user_profile = data.get('user_profile')
        jobs = data.get('jobs', [])
        top_k = data.get('top_k', 10)
        
        if not user_profile:
            return jsonify({
                "success": False,
                "error": "user_profile is required"
            }), 400
        
        if not jobs:
            return jsonify({
                "success": False,
                "error": "jobs list is required"
            }), 400
        
        # Extract user features
        user_features = feature_extractor.extract_user_features(user_profile)
        
        # Extract job features
        job_features_list = []
        for job in jobs:
            job_feat = feature_extractor.extract_job_features(job)
            job_features_list.append(job_feat)
        
        # Prepare batch data
        batch_data = feature_extractor.prepare_batch_data(
            user_features,
            job_features_list,
            device=device
        )
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = []
            batch_size = len(jobs)
            
            for i in range(batch_size):
                pred = model(
                    batch_data['user_text'][i:i+1],
                    batch_data['job_text'][i:i+1],
                    {k: v[i:i+1] for k, v in batch_data['user_categorical'].items()},
                    [batch_data['job_categoricals'][i]],
                    batch_data['additional_features'][i:i+1]
                )
                predictions.append(pred.item())
        
        # Get top-k recommendations
        predictions = np.array(predictions)
        top_k_indices = np.argsort(predictions)[::-1][:top_k]
        
        # Prepare recommendations
        recommendations = []
        for idx in top_k_indices:
            job = jobs[idx].copy()
            job['match_score'] = float(predictions[idx])
            job['rank'] = len(recommendations) + 1
            
            # Add skill matching info
            user_skills = set(s.lower() for s in user_features.get('skills', []))
            job_skills = set(s.lower() for s in job_features_list[idx].get('skills', []))
            matching_skills = list(user_skills.intersection(job_skills))
            
            job['matching_skills'] = matching_skills
            job['skill_match_count'] = len(matching_skills)
            
            recommendations.append(job)
        
        return jsonify({
            "success": True,
            "count": len(recommendations),
            "recommended_jobs": recommendations,
            "user_profile_summary": {
                "skills": user_features.get('skills', []),
                "experience_years": user_features.get('experience_years', 0),
                "preferred_location": user_features.get('preferred_location', 'any')
            },
            "average_match_score": float(np.mean([r['match_score'] for r in recommendations]))
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/scrape-and-recommend', methods=['POST'])
def scrape_and_recommend():
    """
    Scrape jobs and get AI recommendations in one request
    
    Request body:
    {
        "search_term": "python developer",
        "location": "Remote",
        "results_wanted": 20,
        "user_profile": {...},
        "top_k": 10
    }
    """
    try:
        data = request.get_json()
        
        # Scraping parameters
        site_name = data.get('site_name', ['indeed', 'linkedin'])
        search_term = data.get('search_term', 'python developer')
        location = data.get('location', 'United States')
        results_wanted = int(data.get('results_wanted', 20))
        is_remote = data.get('is_remote', False)
        job_type = data.get('job_type', None)
        hours_old = data.get('hours_old', 72)
        
        # User profile and recommendation parameters
        user_profile = data.get('user_profile')
        top_k = data.get('top_k', 10)
        
        if not user_profile:
            return jsonify({
                "success": False,
                "error": "user_profile is required for recommendations"
            }), 400
        
        # Step 1: Scrape jobs
        print(f"Scraping jobs for: {search_term} in {location}")
        jobs_df = scrape_jobs(
            site_name=site_name,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            is_remote=is_remote,
            job_type=job_type,
            hours_old=hours_old,
            linkedin_fetch_description=True,
            description_format='plain',
            verbose=0
        )
        
        jobs_list = jobs_df.to_dict(orient='records')
        
        if not jobs_list:
            return jsonify({
                "success": False,
                "error": "No jobs found matching the criteria"
            }), 404
        
        # Step 2: Get recommendations (if model is loaded)
        if model_loaded:
            # Extract user features
            user_features = feature_extractor.extract_user_features(user_profile)
            
            # Extract job features
            job_features_list = []
            for job in jobs_list:
                job_feat = feature_extractor.extract_job_features(job)
                job_features_list.append(job_feat)
            
            # Prepare batch data
            batch_data = feature_extractor.prepare_batch_data(
                user_features,
                job_features_list,
                device=device
            )
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                predictions = []
                batch_size = len(jobs_list)
                
                for i in range(batch_size):
                    pred = model(
                        batch_data['user_text'][i:i+1],
                        batch_data['job_text'][i:i+1],
                        {k: v[i:i+1] for k, v in batch_data['user_categorical'].items()},
                        [batch_data['job_categoricals'][i]],
                        batch_data['additional_features'][i:i+1]
                    )
                    predictions.append(pred.item())
            
            # Get top-k recommendations
            predictions = np.array(predictions)
            top_k_indices = np.argsort(predictions)[::-1][:top_k]
            
            # Prepare recommendations
            recommendations = []
            for idx in top_k_indices:
                job = jobs_list[idx].copy()
                job['match_score'] = float(predictions[idx])
                job['rank'] = len(recommendations) + 1
                
                # Add skill matching info
                user_skills = set(s.lower() for s in user_features.get('skills', []))
                job_skills = set(s.lower() for s in job_features_list[idx].get('skills', []))
                matching_skills = list(user_skills.intersection(job_skills))
                
                job['matching_skills'] = matching_skills
                job['skill_match_count'] = len(matching_skills)
                
                recommendations.append(job)
            
            avg_match_score = float(np.mean([r['match_score'] for r in recommendations]))
        else:
            # Fallback: simple skill-based matching
            recommendations = simple_skill_based_ranking(user_profile, jobs_list, top_k)
            avg_match_score = 0.0
        
        return jsonify({
            "success": True,
            "scraped_count": len(jobs_list),
            "recommendation_count": len(recommendations),
            "recommended_jobs": recommendations,
            "all_jobs": jobs_list,
            "search_params": {
                "search_term": search_term,
                "location": location,
                "sites": site_name
            },
            "model_used": "ConvFM" if model_loaded else "Rule-based",
            "average_match_score": avg_match_score
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def simple_skill_based_ranking(user_profile: Dict, jobs: List[Dict], top_k: int) -> List[Dict]:
    """
    Fallback skill-based ranking when model is not loaded
    """
    user_skills = set(s.lower() for s in user_profile.get('skills', []))
    
    scored_jobs = []
    for job in jobs:
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()
        
        # Count skill matches
        match_count = sum(1 for skill in user_skills if skill in job_text)
        
        job_copy = job.copy()
        job_copy['match_score'] = match_count
        job_copy['matching_skills'] = [s for s in user_skills if s in job_text]
        scored_jobs.append(job_copy)
    
    # Sort by match score
    scored_jobs.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Add rank
    for i, job in enumerate(scored_jobs[:top_k]):
        job['rank'] = i + 1
    
    return scored_jobs[:top_k]


@app.route('/stats', methods=['GET'])
def stats():
    """Get API statistics"""
    return jsonify({
        "statistics": {
            "api_version": "3.1.0",
            "model_status": "loaded" if model_loaded else "not_loaded",
            "device": device,
            "supported_sites": ["linkedin", "indeed", "glassdoor", "naukri", "bdjobs", "ziprecruiter"],
            "features": [
                "Real-time job scraping",
                "AI-powered recommendations (ConvFM)",
                "Multi-site aggregation",
                "Skill matching",
                "Salary compatibility"
            ]
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 AI Job Aggregator & Recommender API Server Starting...")
    print("=" * 60)
    
    # Load model if available
    load_model_and_extractor()
    
    print("API running at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  GET  /model/info - Model information")
    print("  POST /scrape     - Scrape jobs")
    print("  POST /recommend  - Get AI recommendations")
    print("  POST /scrape-and-recommend - Scrape + Recommend")
    print("  GET  /stats      - Statistics")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)