#!/usr/bin/env python3
"""
Advanced Usage Examples for AI Job Aggregator & Recommender API

This script demonstrates advanced features including batch processing,
custom filtering, data analysis, and integration patterns.
"""

import requests
import json
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

class JobAnalyzer:
    """Advanced job data analysis and processing."""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.session = requests.Session()
    
    def make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict:
        """Make HTTP request with session management."""
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data)
            else:
                response = self.session.post(url, json=data, headers=HEADERS)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def batch_job_search(self, search_terms: List[str], location: str = "Remote") -> Dict[str, Dict]:
        """Perform batch job searches for multiple terms."""
        print(f"🔄 Starting batch search for {len(search_terms)} terms...")
        
        results = {}
        for i, term in enumerate(search_terms, 1):
            print(f"   [{i}/{len(search_terms)}] Searching: {term}")
            
            jobs_data = self.make_request("/scrape", "POST", {
                "search_term": term,
                "location": location,
                "results_wanted": 10,
                "is_remote": True
            })
            
            results[term] = jobs_data
            
            # Rate limiting
            time.sleep(2)
        
        return results
    
    def analyze_salary_trends(self, jobs_data: List[Dict]) -> Dict:
        """Analyze salary trends from job data."""
        print("📊 Analyzing salary trends...")
        
        salaries = []
        for job in jobs_data:
            if job.get('min_amount') and job.get('max_amount'):
                avg_salary = (job['min_amount'] + job['max_amount']) / 2
                salaries.append({
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'salary': avg_salary,
                    'currency': job.get('currency', 'USD'),
                    'site': job['site']
                })
        
        if not salaries:
            return {"error": "No salary data available"}
        
        df = pd.DataFrame(salaries)
        
        analysis = {
            'total_jobs_with_salary': len(salaries),
            'average_salary': df['salary'].mean(),
            'median_salary': df['salary'].median(),
            'salary_range': {
                'min': df['salary'].min(),
                'max': df['salary'].max()
            },
            'top_paying_companies': df.groupby('company')['salary'].mean().sort_values(ascending=False).head(5).to_dict(),
            'salary_by_location': df.groupby('location')['salary'].mean().sort_values(ascending=False).to_dict()
        }
        
        return analysis
    
    def skills_frequency_analysis(self, jobs_data: List[Dict]) -> Dict:
        """Analyze most frequently mentioned skills in job descriptions."""
        print("🔍 Analyzing skills frequency...")
        
        skill_counts = {}
        total_jobs = len(jobs_data)
        
        # Common tech skills to look for
        tech_skills = [
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust',
            'React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask',
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform',
            'SQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
            'Git', 'Jenkins', 'CI/CD', 'Microservices', 'REST', 'GraphQL'
        ]
        
        for job in jobs_data:
            description = job.get('description', '').lower()
            
            for skill in tech_skills:
                if skill.lower() in description:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Calculate percentages
        skill_percentages = {
            skill: (count / total_jobs) * 100 
            for skill, count in skill_counts.items()
        }
        
        # Sort by frequency
        sorted_skills = sorted(skill_percentages.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_jobs_analyzed': total_jobs,
            'top_skills': dict(sorted_skills[:15]),
            'skill_counts': skill_counts
        }
    
    def company_analysis(self, jobs_data: List[Dict]) -> Dict:
        """Analyze companies and job market trends."""
        print("🏢 Analyzing company trends...")
        
        companies = {}
        locations = {}
        job_types = {}
        
        for job in jobs_data:
            # Company analysis
            company = job.get('company', 'Unknown')
            if company not in companies:
                companies[company] = {
                    'count': 0,
                    'locations': set(),
                    'job_types': set(),
                    'avg_salary': []
                }
            
            companies[company]['count'] += 1
            companies[company]['locations'].add(job.get('location', 'Unknown'))
            companies[company]['job_types'].add(job.get('job_type', 'Unknown'))
            
            if job.get('min_amount') and job.get('max_amount'):
                avg_salary = (job['min_amount'] + job['max_amount']) / 2
                companies[company]['avg_salary'].append(avg_salary)
            
            # Location analysis
            location = job.get('location', 'Unknown')
            locations[location] = locations.get(location, 0) + 1
            
            # Job type analysis
            job_type = job.get('job_type', 'Unknown')
            job_types[job_type] = job_types.get(job_type, 0) + 1
        
        # Calculate averages
        for company in companies.values():
            if company['avg_salary']:
                company['avg_salary'] = sum(company['avg_salary']) / len(company['avg_salary'])
            else:
                company['avg_salary'] = None
            
            company['locations'] = list(company['locations'])
            company['job_types'] = list(company['job_types'])
        
        # Top companies by job count
        top_companies = sorted(
            [(name, data['count']) for name, data in companies.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_companies': len(companies),
            'top_companies_by_jobs': dict(top_companies),
            'top_locations': dict(sorted(locations.items(), key=lambda x: x[1], reverse=True)[:10]),
            'job_type_distribution': job_types,
            'company_details': companies
        }

async def async_batch_requests(search_terms: List[str], location: str = "Remote") -> List[Dict]:
    """Perform asynchronous batch requests for better performance."""
    print(f"⚡ Starting async batch requests for {len(search_terms)} terms...")
    
    async def fetch_jobs(session: aiohttp.ClientSession, term: str) -> Dict:
        """Fetch jobs for a single search term."""
        url = f"{API_BASE_URL}/scrape"
        data = {
            "search_term": term,
            "location": location,
            "results_wanted": 5,
            "is_remote": True
        }
        
        try:
            async with session.post(url, json=data, headers=HEADERS) as response:
                result = await response.json()
                return {"term": term, "data": result}
        except Exception as e:
            return {"term": term, "error": str(e)}
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_jobs(session, term) for term in search_terms]
        results = await asyncio.gather(*tasks)
    
    return results

def custom_filtering_example():
    """Demonstrate custom filtering and data processing."""
    print("\n" + "="*60)
    print("🔧 CUSTOM FILTERING & PROCESSING")
    print("="*60)
    
    analyzer = JobAnalyzer()
    
    # Get jobs data
    jobs_data = analyzer.make_request("/scrape", "POST", {
        "search_term": "software engineer",
        "location": "United States",
        "results_wanted": 20,
        "is_remote": True
    })
    
    if "error" in jobs_data:
        print(f"❌ Failed to get jobs data: {jobs_data['error']}")
        return
    
    jobs = jobs_data.get('jobs', [])
    print(f"✅ Retrieved {len(jobs)} jobs for analysis")
    
    # Perform various analyses
    salary_analysis = analyzer.analyze_salary_trends(jobs)
    if "error" not in salary_analysis:
        print(f"\n💰 Salary Analysis:")
        print(f"   Average Salary: ${salary_analysis['average_salary']:,.0f}")
        print(f"   Median Salary: ${salary_analysis['median_salary']:,.0f}")
        print(f"   Salary Range: ${salary_analysis['salary_range']['min']:,.0f} - ${salary_analysis['salary_range']['max']:,.0f}")
        
        print(f"\n   Top Paying Companies:")
        for company, salary in list(salary_analysis['top_paying_companies'].items())[:3]:
            print(f"      {company}: ${salary:,.0f}")
    
    skills_analysis = analyzer.skills_frequency_analysis(jobs)
    print(f"\n🔍 Skills Analysis:")
    print(f"   Total Jobs Analyzed: {skills_analysis['total_jobs_analyzed']}")
    print(f"   Top Skills:")
    for skill, percentage in list(skills_analysis['top_skills'].items())[:5]:
        print(f"      {skill}: {percentage:.1f}%")
    
    company_analysis = analyzer.company_analysis(jobs)
    print(f"\n🏢 Company Analysis:")
    print(f"   Total Companies: {company_analysis['total_companies']}")
    print(f"   Top Companies by Job Count:")
    for company, count in list(company_analysis['top_companies_by_jobs'].items())[:5]:
        print(f"      {company}: {count} jobs")

def batch_processing_example():
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*60)
    print("📦 BATCH PROCESSING")
    print("="*60)
    
    analyzer = JobAnalyzer()
    
    # Define search terms for batch processing
    search_terms = [
        "python developer",
        "machine learning engineer",
        "data scientist",
        "backend developer",
        "devops engineer"
    ]
    
    # Perform batch search
    batch_results = analyzer.batch_job_search(search_terms, "Remote")
    
    # Analyze results
    total_jobs = 0
    successful_searches = 0
    
    print(f"\n📊 Batch Processing Results:")
    for term, result in batch_results.items():
        if "error" not in result:
            count = result.get('count', 0)
            total_jobs += count
            successful_searches += 1
            print(f"   ✅ {term}: {count} jobs")
        else:
            print(f"   ❌ {term}: {result['error']}")
    
    print(f"\n📈 Summary:")
    print(f"   Successful Searches: {successful_searches}/{len(search_terms)}")
    print(f"   Total Jobs Found: {total_jobs}")

async def async_processing_example():
    """Demonstrate asynchronous processing."""
    print("\n" + "="*60)
    print("⚡ ASYNC PROCESSING")
    print("="*60)
    
    search_terms = [
        "frontend developer",
        "mobile developer",
        "cloud engineer",
        "security engineer"
    ]
    
    start_time = time.time()
    async_results = await async_batch_requests(search_terms)
    end_time = time.time()
    
    print(f"✅ Async processing completed in {end_time - start_time:.2f} seconds")
    
    total_jobs = 0
    for result in async_results:
        if "error" not in result:
            term = result['term']
            count = result['data'].get('count', 0)
            total_jobs += count
            print(f"   {term}: {count} jobs")
        else:
            print(f"   {result['term']}: Error - {result['error']}")
    
    print(f"\n📊 Total jobs found: {total_jobs}")

def data_export_examples():
    """Demonstrate data export capabilities."""
    print("\n" + "="*60)
    print("💾 DATA EXPORT EXAMPLES")
    print("="*60)
    
    analyzer = JobAnalyzer()
    
    # Get jobs data
    jobs_data = analyzer.make_request("/scrape", "POST", {
        "search_term": "data analyst",
        "location": "Remote",
        "results_wanted": 15
    })
    
    if "error" in jobs_data:
        print(f"❌ Failed to get jobs data: {jobs_data['error']}")
        return
    
    jobs = jobs_data.get('jobs', [])
    
    # Export to different formats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON export
    json_filename = f"jobs_export_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(jobs_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Exported to JSON: {json_filename}")
    
    # CSV export using pandas
    if jobs:
        df = pd.DataFrame(jobs)
        csv_filename = f"jobs_export_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Exported to CSV: {csv_filename}")
        
        # Create summary report
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_jobs': len(jobs),
            'sites': df['site'].value_counts().to_dict(),
            'companies': df['company'].value_counts().head(10).to_dict(),
            'locations': df['location'].value_counts().head(10).to_dict()
        }
        
        summary_filename = f"jobs_summary_{timestamp}.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Created summary report: {summary_filename}")

async def main():
    """Main function to run all advanced examples."""
    print("🚀 AI Job Aggregator - Advanced Usage Examples")
    print("="*80)
    
    # Check API health
    analyzer = JobAnalyzer()
    health = analyzer.make_request("/health")
    if "error" in health:
        print("❌ API is not responding. Please start the server first.")
        return
    
    print("✅ API is healthy, starting advanced examples...")
    
    # Run examples
    custom_filtering_example()
    batch_processing_example()
    await async_processing_example()
    data_export_examples()
    
    print("\n" + "="*80)
    print("✅ All advanced examples completed!")
    print("📚 Check the generated files for exported data")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
