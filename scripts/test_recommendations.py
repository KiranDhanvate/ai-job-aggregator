#!/usr/bin/env python3
"""
Test Script for Job Recommendations

This script tests the recommendation system with various scenarios and metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.convfm_model import ConvFMJobRecommender
from ml_models.feature_extractor import JobFeatureExtractor
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationTester:
    """
    Test suite for job recommendation system
    """
    
    def __init__(self, model_path: str = None, data_path: str = "data"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.feature_extractor = None
        self.jobs_df = None
        self.interactions_df = None
        
        # Test results
        self.test_results = {}
    
    def load_model_and_data(self):
        """Load trained model and test data"""
        logger.info("Loading model and data...")
        
        # Load job data
        jobs_file = os.path.join(self.data_path, "collected_jobs.csv")
        if os.path.exists(jobs_file):
            self.jobs_df = pd.read_csv(jobs_file)
            logger.info(f"Loaded {len(self.jobs_df)} jobs")
        else:
            logger.error(f"Jobs file not found: {jobs_file}")
            return False
        
        # Load interactions
        interactions_file = os.path.join(self.data_path, "user_interactions.csv")
        if os.path.exists(interactions_file):
            self.interactions_df = pd.read_csv(interactions_file)
            logger.info(f"Loaded {len(self.interactions_df)} interactions")
        else:
            logger.error(f"Interactions file not found: {interactions_file}")
            return False
        
        # Load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        
        return True
    
    def _load_model(self):
        """Load trained model"""
        try:
            import torch
            
            # Find config file
            config_files = [f for f in os.listdir('artifacts') if f.endswith('_config.json')]
            if not config_files:
                logger.error("No model config found")
                return
            
            config_path = os.path.join('artifacts', config_files[0])
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load feature extractor
            extractor_files = [f for f in os.listdir('artifacts') if f.endswith('_extractor.pkl')]
            if extractor_files:
                extractor_path = os.path.join('artifacts', extractor_files[0])
                self.feature_extractor = JobFeatureExtractor.load(extractor_path)
            else:
                logger.error("No feature extractor found")
                return
            
            # Initialize model
            model_config = {
                'num_users': len(self.feature_extractor.user_encoder.classes_),
                'num_jobs': len(self.feature_extractor.job_encoder.classes_),
                'num_skills': len(self.feature_extractor.skill_encoder.classes_),
                'num_locations': len(self.feature_extractor.location_encoder.classes_),
                'num_companies': len(self.feature_extractor.company_encoder.classes_),
                'embedding_dim': config.get('embedding_dim', 64),
                'conv_filters': config.get('conv_filters', 64),
                'conv_kernel_size': config.get('conv_kernel_size', 3),
                'dropout_rate': config.get('dropout_rate', 0.2),
                'hidden_dims': config.get('hidden_dims', [128, 64, 32])
            }
            
            self.model = ConvFMJobRecommender(**model_config)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def test_basic_recommendations(self, num_users: int = 10, top_k: int = 10) -> Dict[str, Any]:
        """
        Test basic recommendation generation
        
        Args:
            num_users: Number of users to test
            top_k: Number of recommendations per user
            
        Returns:
            Test results dictionary
        """
        logger.info(f"Testing basic recommendations for {num_users} users...")
        
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        results = {
            'total_users_tested': num_users,
            'top_k': top_k,
            'recommendations_per_user': [],
            'avg_recommendations': 0,
            'success_rate': 0
        }
        
        successful_users = 0
        
        for user_id in range(num_users):
            try:
                # Get user's actual interactions
                user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
                
                if len(user_interactions) == 0:
                    logger.warning(f"No interactions found for user {user_id}")
                    continue
                
                # Generate recommendations
                recommendations = self._generate_recommendations_for_user(user_id, top_k)
                
                results['recommendations_per_user'].append(len(recommendations))
                
                if len(recommendations) > 0:
                    successful_users += 1
                
                logger.info(f"User {user_id}: {len(recommendations)} recommendations")
                
            except Exception as e:
                logger.error(f"Error testing user {user_id}: {str(e)}")
                results['recommendations_per_user'].append(0)
        
        results['avg_recommendations'] = np.mean(results['recommendations_per_user'])
        results['success_rate'] = successful_users / num_users
        
        logger.info(f"Basic test completed. Success rate: {results['success_rate']:.2%}")
        
        return results
    
    def test_recommendation_quality(self, num_users: int = 20, top_k: int = 10) -> Dict[str, Any]:
        """
        Test recommendation quality using various metrics
        
        Args:
            num_users: Number of users to test
            top_k: Number of recommendations per user
            
        Returns:
            Quality test results
        """
        logger.info(f"Testing recommendation quality for {num_users} users...")
        
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        results = {
            'total_users_tested': num_users,
            'top_k': top_k,
            'metrics': {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'ndcg': []
            },
            'avg_metrics': {}
        }
        
        for user_id in range(num_users):
            try:
                # Get user's actual interactions
                user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
                
                if len(user_interactions) < 2:  # Need at least 2 interactions for meaningful test
                    continue
                
                # Split user interactions into train/test
                train_interactions = user_interactions.sample(frac=0.7)
                test_interactions = user_interactions.drop(train_interactions.index)
                
                # Generate recommendations based on train interactions
                recommendations = self._generate_recommendations_for_user(user_id, top_k)
                recommended_job_ids = [rec['job_id'] for rec in recommendations]
                
                # Get test job IDs (ground truth)
                test_job_ids = test_interactions['job_id'].tolist()
                
                # Calculate metrics
                metrics = self._calculate_recommendation_metrics(
                    recommended_job_ids, test_job_ids, top_k
                )
                
                for metric, value in metrics.items():
                    results['metrics'][metric].append(value)
                
                logger.info(f"User {user_id} metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error testing quality for user {user_id}: {str(e)}")
                continue
        
        # Calculate average metrics
        for metric in results['metrics']:
            if results['metrics'][metric]:
                results['avg_metrics'][metric] = np.mean(results['metrics'][metric])
            else:
                results['avg_metrics'][metric] = 0.0
        
        logger.info(f"Quality test completed. Avg metrics: {results['avg_metrics']}")
        
        return results
    
    def test_api_endpoints(self, api_url: str = "http://localhost:8001") -> Dict[str, Any]:
        """
        Test API endpoints
        
        Args:
            api_url: Base URL of the API server
            
        Returns:
            API test results
        """
        logger.info(f"Testing API endpoints at {api_url}...")
        
        results = {
            'endpoints_tested': [],
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': []
        }
        
        # Test endpoints
        endpoints = [
            {"path": "/", "method": "GET"},
            {"path": "/health", "method": "GET"},
            {"path": "/model_info", "method": "GET"},
            {"path": "/recommend", "method": "POST", "data": {
                "user_id": 1,
                "max_recommendations": 5
            }},
            {"path": "/recommendations/1", "method": "GET"}
        ]
        
        for endpoint in endpoints:
            try:
                start_time = datetime.now()
                
                if endpoint["method"] == "GET":
                    response = requests.get(f"{api_url}{endpoint['path']}", timeout=10)
                else:
                    response = requests.post(
                        f"{api_url}{endpoint['path']}",
                        json=endpoint.get("data", {}),
                        timeout=10
                    )
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                endpoint_result = {
                    "path": endpoint["path"],
                    "method": endpoint["method"],
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "success": response.status_code == 200
                }
                
                results['endpoints_tested'].append(endpoint_result)
                results['response_times'].append(response_time)
                
                if endpoint_result['success']:
                    results['successful_requests'] += 1
                else:
                    results['failed_requests'] += 1
                    logger.warning(f"Endpoint {endpoint['path']} failed with status {response.status_code}")
                
                logger.info(f"Endpoint {endpoint['path']}: {response.status_code} ({response_time:.1f}ms)")
                
            except Exception as e:
                logger.error(f"Error testing endpoint {endpoint['path']}: {str(e)}")
                results['failed_requests'] += 1
        
        results['avg_response_time'] = np.mean(results['response_times']) if results['response_times'] else 0
        
        logger.info(f"API test completed. {results['successful_requests']} successful, {results['failed_requests']} failed")
        
        return results
    
    def test_performance(self, num_requests: int = 100) -> Dict[str, Any]:
        """
        Test system performance under load
        
        Args:
            num_requests: Number of requests to simulate
            
        Returns:
            Performance test results
        """
        logger.info(f"Testing performance with {num_requests} requests...")
        
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        results = {
            'total_requests': num_requests,
            'response_times': [],
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        # Generate test users
        test_users = np.random.choice(
            self.interactions_df['user_id'].unique(),
            size=num_requests,
            replace=True
        )
        
        for i, user_id in enumerate(test_users):
            try:
                start_time = datetime.now()
                
                # Generate recommendations
                recommendations = self._generate_recommendations_for_user(user_id, top_k=10)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                results['response_times'].append(response_time)
                results['successful_requests'] += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{num_requests} requests")
                
            except Exception as e:
                logger.error(f"Error in performance test request {i}: {str(e)}")
                results['failed_requests'] += 1
        
        # Calculate performance metrics
        if results['response_times']:
            results['avg_response_time'] = np.mean(results['response_times'])
            results['median_response_time'] = np.median(results['response_times'])
            results['p95_response_time'] = np.percentile(results['response_times'], 95)
            results['p99_response_time'] = np.percentile(results['response_times'], 99)
        
        results['requests_per_second'] = results['successful_requests'] / (sum(results['response_times']) / 1000) if results['response_times'] else 0
        
        logger.info(f"Performance test completed. Avg response time: {results.get('avg_response_time', 0):.1f}ms")
        
        return results
    
    def _generate_recommendations_for_user(self, user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Generate recommendations for a user"""
        # Get all job IDs
        job_ids = self.jobs_df['job_id'].tolist()
        
        # Generate user profile (simplified)
        user_skills = []
        location_id = 0
        company_id = 0
        
        # Get model predictions
        scores = self.model.predict_job_scores(
            user_id=user_id,
            job_ids=job_ids,
            skill_ids=user_skills,
            location_id=location_id,
            company_id=company_id
        )
        
        # Sort by score and get top recommendations
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format recommendations
        recommendations = []
        for job_id, score in sorted_scores[:top_k]:
            job_data = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
            
            recommendation = {
                'job_id': int(job_id),
                'title': job_data['title'],
                'company': job_data['company'],
                'location': job_data['location'],
                'score': float(score)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_recommendation_metrics(self, recommended: List[int], relevant: List[int], k: int) -> Dict[str, float]:
        """Calculate recommendation quality metrics"""
        metrics = {}
        
        # Convert to binary vectors
        recommended_binary = np.zeros(len(recommended))
        relevant_binary = np.zeros(len(recommended))
        
        for i, job_id in enumerate(recommended):
            if job_id in relevant:
                relevant_binary[i] = 1
        
        # Precision@K
        if len(recommended) > 0:
            metrics['precision'] = np.sum(relevant_binary) / min(k, len(recommended))
        else:
            metrics['precision'] = 0.0
        
        # Recall@K
        if len(relevant) > 0:
            metrics['recall'] = np.sum(relevant_binary) / len(relevant)
        else:
            metrics['recall'] = 0.0
        
        # F1@K
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        # NDCG@K
        try:
            metrics['ndcg'] = ndcg_score([relevant_binary], [np.ones(len(recommended))], k=k)
        except:
            metrics['ndcg'] = 0.0
        
        return metrics
    
    def run_all_tests(self, api_url: str = None) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results
        
        Args:
            api_url: API URL for testing (optional)
            
        Returns:
            Complete test results
        """
        logger.info("Running comprehensive test suite...")
        
        if not self.load_model_and_data():
            logger.error("Failed to load model and data")
            return {}
        
        all_results = {
            'test_timestamp': datetime.now().isoformat(),
            'basic_recommendations': {},
            'quality_metrics': {},
            'performance': {},
            'api_tests': {}
        }
        
        # Test basic recommendations
        all_results['basic_recommendations'] = self.test_basic_recommendations()
        
        # Test recommendation quality
        all_results['quality_metrics'] = self.test_recommendation_quality()
        
        # Test performance
        all_results['performance'] = self.test_performance(num_requests=50)
        
        # Test API if URL provided
        if api_url:
            all_results['api_tests'] = self.test_api_endpoints(api_url)
        
        # Save results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"All tests completed. Results saved to {results_file}")
        
        return all_results


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Job Recommendation System')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to test data directory')
    parser.add_argument('--api_url', type=str, default=None,
                       help='API URL for testing')
    parser.add_argument('--test_type', type=str, choices=['basic', 'quality', 'performance', 'api', 'all'],
                       default='all', help='Type of test to run')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RecommendationTester(args.model_path, args.data_path)
    
    try:
        if args.test_type == 'all':
            results = tester.run_all_tests(args.api_url)
        else:
            if not tester.load_model_and_data():
                logger.error("Failed to load model and data")
                return 1
            
            if args.test_type == 'basic':
                results = tester.test_basic_recommendations()
            elif args.test_type == 'quality':
                results = tester.test_recommendation_quality()
            elif args.test_type == 'performance':
                results = tester.test_performance()
            elif args.test_type == 'api':
                if not args.api_url:
                    logger.error("API URL required for API testing")
                    return 1
                results = tester.test_api_endpoints(args.api_url)
        
        # Print summary
        print("\n" + "="*50)
        print("TEST RESULTS SUMMARY")
        print("="*50)
        
        if args.test_type == 'all':
            print(f"Basic Recommendations: {results['basic_recommendations'].get('success_rate', 0):.2%} success rate")
            print(f"Quality Metrics: {results['quality_metrics'].get('avg_metrics', {})}")
            print(f"Performance: {results['performance'].get('avg_response_time', 0):.1f}ms avg response time")
            if results['api_tests']:
                print(f"API Tests: {results['api_tests']['successful_requests']} successful requests")
        else:
            print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
