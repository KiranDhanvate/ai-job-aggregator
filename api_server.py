from flask import Flask, request, jsonify
from flask_cors import CORS
from jobspy import scrape_jobs
import pandas as pd
import re
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def clean_description(html_text):
    """Remove HTML tags and clean text"""
    if not html_text or pd.isna(html_text):
        return ""
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
        "message": "JobSpy API is running!",
        "endpoints": {
            "/scrape": "POST - Scrape jobs",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "JobSpy API"})

@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        # Get parameters from request
        data = request.get_json()
        
        site_name = data.get('site_name', ['indeed', 'linkedin'])
        search_term = data.get('search_term', 'python developer')
        location = data.get('location', 'United States')
        results_wanted = data.get('results_wanted', 10)
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
            verbose=0  # Disable logging for API
        )
        
        # Convert DataFrame to JSON
        jobs_list = jobs_df.to_dict(orient='records')
        
        # Clean descriptions if requested
        for job in jobs_list:
            if 'description' in job and description_format == 'plain':
                job['description_clean'] = clean_description(job['description'])
        
        return jsonify({
            "success": True,
            "count": len(jobs_list),
            "jobs": jobs_list
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/scrape', methods=['GET'])
def scrape_get():
    try:
        # Get parameters from query string
        site_name = request.args.get('site_name', 'indeed').split(',')
        search_term = request.args.get('search_term', 'python developer')
        location = request.args.get('location', 'USA')
        results_wanted = int(request.args.get('results_wanted', 10))
        
        # Scrape jobs
        jobs_df = scrape_jobs(
            site_name=site_name,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            verbose=0
        )
        
        # Convert to JSON
        jobs_list = jobs_df.to_dict(orient='records')
        
        return jsonify({
            "success": True,
            "count": len(jobs_list),
            "jobs": jobs_list
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 JobSpy API Server Starting...")
    print("="*60)
    print("API running at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  GET  /scrape     - Scrape jobs (query params)")
    print("  POST /scrape     - Scrape jobs (JSON body)")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)