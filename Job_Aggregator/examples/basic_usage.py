#!/usr/bin/env python3
"""
Basic Usage Examples for AI Job Aggregator & Recommender API

This script demonstrates how to use the API for various job scraping
and recommendation scenarios.
"""

import requests
import json
import time
from typing import Dict, List, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def make_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
    """Make HTTP request to API with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=data)
        else:
            response = requests.post(url, json=data, headers=HEADERS)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request to {endpoint}: {e}")
        return {"error": str(e)}

def check_api_health():
    """Check if the API is running and healthy."""
    print("🔍 Checking API health...")
    health_data = make_request("/health")
    
    if "error" not in health_data:
        print("✅ API is healthy!")
        print(f"   Status: {health_data.get('status', 'unknown')}")
        print(f"   Service: {health_data.get('service', 'unknown')}")
        print(f"   Mode: {health_data.get('mode', 'unknown')}")
        return True
    else:
        print("❌ API is not responding")
        return False

def basic_job_scraping():
    """Demonstrate basic job scraping functionality."""
    print("\n" + "="*60)
    print("🔍 BASIC JOB SCRAPING")
    print("="*60)
    
    # Example 1: Simple job search
    print("\n📋 Example 1: Simple Python Developer Search")
    jobs_data = make_request("/scrape", "POST", {
        "search_term": "python developer",
        "location": "San Francisco",
        "results_wanted": 5,
        "is_remote": False
    })
    
    if "error" not in jobs_data:
        print(f"✅ Found {jobs_data['count']} jobs")
        for i, job in enumerate(jobs_data['jobs'][:3], 1):
            print(f"   {i}. {job['title']} at {job['company']}")
            print(f"      Location: {job['location']}")
            print(f"      Posted: {job['date_posted']}")
    else:
        print(f"❌ Scraping failed: {jobs_data['error']}")
    
    # Example 2: Remote jobs with filters
    print("\n📋 Example 2: Remote Machine Learning Jobs")
    remote_jobs = make_request("/scrape", "POST", {
        "search_term": "machine learning engineer",
        "location": "Remote",
        "results_wanted": 3,
        "is_remote": True,
        "hours_old": 72,  # Jobs posted in last 3 days
        "job_type": "fulltime"
    })
    
    if "error" not in remote_jobs:
        print(f"✅ Found {remote_jobs['count']} remote ML jobs")
        for i, job in enumerate(remote_jobs['jobs'][:2], 1):
            print(f"   {i}. {job['title']} at {job['company']}")
            print(f"      Remote: {job['is_remote']}")
            print(f"      Type: {job['job_type']}")
    else:
        print(f"❌ Remote job search failed: {remote_jobs['error']}")

def ai_recommendations():
    """Demonstrate AI-powered job recommendations."""
    print("\n" + "="*60)
    print("🤖 AI-POWERED RECOMMENDATIONS")
    print("="*60)
    
    # Example: Get personalized recommendations
    print("\n🎯 Example: Personalized Job Recommendations")
    recommendations = make_request("/scrape-and-recommend", "POST", {
        "search_term": "data scientist",
        "location": "New York",
        "user_skills": ["Python", "TensorFlow", "AWS", "Docker", "SQL"],
        "experience_years": 3,
        "preferred_salary_min": 90000,
        "results_wanted": 5
    })
    
    if "error" not in recommendations:
        print(f"✅ AI found {recommendations['count']} matching jobs")
        print(f"   Match Score: {recommendations.get('average_match_score', 'N/A')}")
        
        for i, job in enumerate(recommendations['recommended_jobs'][:3], 1):
            print(f"\n   {i}. {job['title']} at {job['company']}")
            print(f"      Match Score: {job.get('match_score', 'N/A')}")
            print(f"      Salary: {job.get('salary_source', 'Not specified')}")
            print(f"      Skills Match: {len(job.get('matching_skills', []))} skills")
    else:
        print(f"❌ AI recommendations failed: {recommendations['error']}")

def advanced_filtering():
    """Demonstrate advanced filtering options."""
    print("\n" + "="*60)
    print("🔧 ADVANCED FILTERING")
    print("="*60)
    
    # Example 1: Multiple job sites
    print("\n📋 Example 1: Multi-site Search")
    multi_site_jobs = make_request("/scrape", "POST", {
        "search_term": "frontend developer",
        "location": "Seattle",
        "site_name": ["linkedin", "indeed", "glassdoor"],
        "results_wanted": 8,
        "job_type": "fulltime",
        "hours_old": 48
    })
    
    if "error" not in multi_site_jobs:
        print(f"✅ Found {multi_site_jobs['count']} jobs across multiple sites")
        sites = {}
        for job in multi_site_jobs['jobs']:
            site = job.get('site', 'unknown')
            sites[site] = sites.get(site, 0) + 1
        
        print("   Jobs by site:")
        for site, count in sites.items():
            print(f"      {site}: {count} jobs")
    else:
        print(f"❌ Multi-site search failed: {multi_site_jobs['error']}")
    
    # Example 2: Specific company search
    print("\n📋 Example 2: Company-specific Search")
    company_jobs = make_request("/scrape", "POST", {
        "search_term": "software engineer",
        "location": "United States",
        "results_wanted": 10,
        "linkedin_company_ids": [1441, 1035]  # Example: Microsoft, Apple
    })
    
    if "error" not in company_jobs:
        print(f"✅ Found {company_jobs['count']} jobs at target companies")
        companies = {}
        for job in company_jobs['jobs']:
            company = job.get('company', 'unknown')
            companies[company] = companies.get(company, 0) + 1
        
        print("   Jobs by company:")
        for company, count in companies.items():
            print(f"      {company}: {count} jobs")
    else:
        print(f"❌ Company search failed: {company_jobs['error']}")

def get_api_statistics():
    """Display API usage statistics."""
    print("\n" + "="*60)
    print("📊 API STATISTICS")
    print("="*60)
    
    stats = make_request("/stats")
    
    if "error" not in stats:
        print("✅ API Statistics:")
        statistics = stats.get('statistics', {})
        
        print(f"   Total API Requests: {statistics.get('total_api_requests', 0)}")
        print(f"   Total Jobs Scraped: {statistics.get('total_jobs_scraped', 0)}")
        print(f"   Last Scrape Time: {statistics.get('last_scrape_time', 'Never')}")
        print(f"   Data Source: {statistics.get('data_source', 'Unknown')}")
        print(f"   Cache Status: {statistics.get('cache_status', 'Unknown')}")
        print(f"   Vector DB Status: {statistics.get('vector_db_status', 'Unknown')}")
        
        # Jobs by site breakdown
        jobs_by_site = statistics.get('jobs_by_site', {})
        if jobs_by_site:
            print("\n   Jobs by Site:")
            for site, count in jobs_by_site.items():
                print(f"      {site}: {count} jobs")
    else:
        print(f"❌ Could not retrieve statistics: {stats['error']}")

def export_jobs_to_file(jobs_data: Dict, filename: str):
    """Export jobs data to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(jobs_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Jobs exported to {filename}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def main():
    """Main function to run all examples."""
    print("🚀 AI Job Aggregator & Recommender API - Usage Examples")
    print("="*80)
    
    # Check API health first
    if not check_api_health():
        print("\n❌ Please ensure the API server is running:")
        print("   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Run examples
    basic_job_scraping()
    ai_recommendations()
    advanced_filtering()
    get_api_statistics()
    
    # Example: Export jobs to file
    print("\n" + "="*60)
    print("💾 EXPORT EXAMPLE")
    print("="*60)
    
    export_jobs = make_request("/scrape", "POST", {
        "search_term": "backend developer",
        "location": "Remote",
        "results_wanted": 5
    })
    
    if "error" not in export_jobs:
        export_jobs_to_file(export_jobs, "example_jobs.json")
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("📚 For more information, visit: http://localhost:8000/docs")
    print("="*80)

if __name__ == "__main__":
    main()
