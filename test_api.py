#!/usr/bin/env python3
"""
Test script for the recommendation API
"""

import sys
import os
sys.path.append('.')

def test_api():
    """Test the API components"""
    print("Testing API components...")
    
    # Test imports
    try:
        from api.recommendation_api import RecommendationAPI
        print("[OK] API import successful")
    except Exception as e:
        print(f"[ERROR] API import failed: {e}")
        return False
    
    try:
        from ml_models.convfm_model import ConvFMJobRecommender
        print("[OK] Model import successful")
    except Exception as e:
        print(f"[ERROR] Model import failed: {e}")
        return False
    
    # Check model files
    model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
    if not model_files:
        print("[ERROR] No model files found")
        return False
    
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
    model_path = os.path.join('models', latest_model)
    print(f"[OK] Using model: {latest_model}")
    
    # Check config files
    config_files = [f for f in os.listdir('artifacts') if f.endswith('_config.json')]
    if not config_files:
        print("[ERROR] No config files found")
        return False
    
    config_path = os.path.join('artifacts', config_files[0])
    print(f"[OK] Using config: {config_files[0]}")
    
    # Test model loading
    try:
        import torch
        model_state = torch.load(model_path, map_location='cpu')
        print("[OK] Model state loaded successfully")
        model_keys = list(model_state.keys())
        print(f"  Model has {len(model_keys)} parameters")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False
    
    # Test API creation
    try:
        api = RecommendationAPI(model_path, config_path)
        print("[OK] API created successfully")
        print(f"  Model loaded: {api.model_loaded}")
        if hasattr(api, 'jobs_df') and api.jobs_df is not None:
            print(f"  Jobs loaded: {len(api.jobs_df)} jobs")
        else:
            print("  No job data loaded")
    except Exception as e:
        print(f"[ERROR] API creation failed: {e}")
        return False
    
    print("[OK] All tests passed! Ready to start API.")
    return True

def start_api():
    """Start the API server"""
    print("\nStarting API server...")
    
    try:
        from api.recommendation_api import RecommendationAPI
        import uvicorn
        
        # Find model and config
        model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join('models', x)))
        model_path = os.path.join('models', latest_model)
        
        config_files = [f for f in os.listdir('artifacts') if f.endswith('_config.json')]
        config_path = os.path.join('artifacts', config_files[0])
        
        # Create API
        api = RecommendationAPI(model_path, config_path)
        
        print("API server starting on http://localhost:8001")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /model_info - Model information")
        print("  POST /recommend - Get job recommendations")
        print("  GET  /docs - API documentation")
        
        # Start server
        uvicorn.run(api.app, host='0.0.0.0', port=8001, log_level='info')
        
    except Exception as e:
        print(f"Failed to start API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "start":
        start_api()
    else:
        test_api()
