# # api_server_realtime.py - Real-Time API with NO caching/vector data
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from jobspy import scrape_jobs
# We will import the prediction function inside the lifespan event to ensure
# the server can start even if the model files are not yet created.

# --- 1. Logging and App Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. Lifespan Event for Model Loading ---
# This special function runs code on application startup and shutdown.
# We load the ML model here to make it available to our endpoints.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server starting up... ---")
    try:
        from recommendation_model.predict import get_recommendations_for_scraped_jobs
        # Make the function available globally within the app state
        app.state.get_recommendations = get_recommendations_for_scraped_jobs
        print("✅ Recommendation model loaded and ready.")
    except Exception as e:
        app.state.get_recommendations = None
        print(f"⚠️ WARNING: Could not load recommendation model: {e}")
        print("   The /scrape-and-recommend endpoint will not work until the model is trained.")
    yield
    # Code below yield runs on shutdown (e.g., app.state.model.clear_session())
    print("--- Server shutting down. ---")


# --- 3. App Initialization with Lifespan ---
app = FastAPI(
    title="AI Job Aggregator & Recommender API",
    description="Combines real-time job scraping with personalized AI-powered recommendations.",
    version="3.1.0",
    lifespan=lifespan # This links our startup/shutdown logic
)

# --- 4. State Management and Pydantic Models ---
api_stats = {
    "total_requests": 0,
    "total_jobs_scraped": 0,
    "last_scrape_time": None
}

class ScrapeRequest(BaseModel):
    search_term: str
    location: str
    results_wanted: int = Field(default=20, gt=0, le=50)
    site_names: List[str] = Field(default=["linkedin"])
    is_remote: bool = False
    job_type: Optional[str] = None
    hours_old: Optional[int] = None
    country_indeed: str = 'usa'

# --- 5. API Endpoints ---

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "🔴 JobSpy Real-Time API with AI Recommendations",
        "version": "3.1-FASTAPI",
        "endpoints": {
            "/scrape-and-recommend/{user_id}": "POST - Scrape jobs and get real-time recommendations.",
            "/health": "GET - Check API health and metrics.",
            "/stats": "GET - Get scraping statistics."
        },
        "model_status": "Loaded" if app.state.get_recommendations else "Not Loaded"
    }

@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status": "healthy",
        "service": "JobSpy Real-Time API",
        "mode": "LIVE_SCRAPING",
        "metrics": api_stats
    }

@app.get("/stats", tags=["Monitoring"])
def stats():
    return {"statistics": api_stats}

@app.post("/scrape-and-recommend/{user_id}", tags=["Core Functionality"])
async def scrape_and_recommend(user_id: int, scrape_params: ScrapeRequest = Body(...)):
    if not app.state.get_recommendations:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation service is unavailable. Please train the model first."
        )

    api_stats["total_requests"] += 1
    scrape_start_time = datetime.now()
    
    logger.info(f"🔴 NEW REAL-TIME REQUEST for user {user_id}")
    logger.info(f"   Search: '{scrape_params.search_term}' in '{scrape_params.location}' on sites {scrape_params.site_names}")

    try:
        jobs_df: pd.DataFrame = await run_in_threadpool(
            scrape_jobs,
            site_name=scrape_params.site_names,
            search_term=scrape_params.search_term,
            location=scrape_params.location,
            results_wanted=scrape_params.results_wanted,
            is_remote=scrape_params.is_remote,
            job_type=scrape_params.job_type,
            hours_old=scrape_params.hours_old,
            country_indeed=scrape_params.country_indeed,
        )
        
        scrape_duration = (datetime.now() - scrape_start_time).total_seconds()

        if jobs_df.empty:
            raise HTTPException(status_code=404, detail=f"No jobs found for '{scrape_params.search_term}'.")

        num_jobs_scraped = len(jobs_df)
        api_stats["total_jobs_scraped"] += num_jobs_scraped
        api_stats["last_scrape_time"] = datetime.now()
        logger.info(f"✅ Scraped {num_jobs_scraped} jobs in {scrape_duration:.2f}s.")

        jobs_json = {"count": num_jobs_scraped, "jobs": jobs_df.to_dict(orient='records')}

        logger.info(f"🤖 Generating recommendations for user_id: {user_id}...")
        recommendations = app.state.get_recommendations(
            user_id=user_id, 
            scraped_jobs_json=jobs_json,
            top_k=10
        )
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
            
        return {
            "user_id": user_id, 
            "search_query": scrape_params.search_term,
            "jobs_scraped": num_jobs_scraped,
            "recommendations": recommendations,
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")



# To run this server, use the following command in your terminal:
# uvicorn api_server:app --reload













# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from jobspy import scrape_jobs
# import pandas as pd
# import re
# from bs4 import BeautifulSoup
# from datetime import datetime
# import logging

# app = Flask(__name__)
# CORS(app)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Metrics
# total_requests = 0
# total_jobs_scraped = 0
# last_scrape_time = None


# def clean_description(html_text):
#     """Remove HTML tags and clean text"""
#     if not html_text or pd.isna(html_text):
#         return ""
#     soup = BeautifulSoup(str(html_text), 'html.parser')
#     text = soup.get_text(separator=' ')
#     text = text.replace('&nbsp;', ' ').replace('&rsquo;', "'")
#     text = text.replace('&ldquo;', '"').replace('&rdquo;', '"')
#     text = text.replace('&ndash;', '-').replace('&mdash;', '—')
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()


# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         "message": "🔴 JobSpy Real-Time API",
#         "version": "3.0-REALTIME",
#         "mode": "LIVE SCRAPING - NO CACHE/VECTOR DATA",
#         "endpoints": {
#             "/scrape": "POST - Scrape jobs in REAL-TIME",
#             "/health": "GET - Check API health",
#             "/stats": "GET - Get scraping statistics"
#         },
#         "features": [
#             "✅ Always fetches LIVE data",
#             "✅ No cached or vector database data",
#             "✅ Real-time job descriptions",
#             "✅ Latest job postings",
#             "✅ Full job details"
#         ],
#         "warning": "⚠️ Real-time scraping may be slower but ensures fresh data"
#     })


# @app.route('/health', methods=['GET'])
# def health():
#     global last_scrape_time, total_requests, total_jobs_scraped
    
#     return jsonify({
#         "status": "healthy",
#         "service": "JobSpy Real-Time API",
#         "mode": "LIVE_SCRAPING",
#         "metrics": {
#             "total_requests": total_requests,
#             "total_jobs_scraped": total_jobs_scraped,
#             "last_scrape": last_scrape_time.isoformat() if last_scrape_time else None,
#             "cache_enabled": False,
#             "vector_db_enabled": False
#         }
#     }), 200


# @app.route('/stats', methods=['GET'])
# def stats():
#     global total_requests, total_jobs_scraped, last_scrape_time
    
#     return jsonify({
#         "statistics": {
#             "total_api_requests": total_requests,
#             "total_jobs_scraped": total_jobs_scraped,
#             "last_scrape_time": last_scrape_time.isoformat() if last_scrape_time else None,
#             "data_source": "LIVE_SCRAPING",
#             "cache_status": "DISABLED",
#             "vector_db_status": "DISABLED"
#         }
#     })


# @app.route('/scrape', methods=['POST'])
# def scrape():
#     global total_requests, total_jobs_scraped, last_scrape_time
    
#     total_requests += 1
#     scrape_start_time = datetime.now()
    
#     logger.info("🔴 NEW REAL-TIME SCRAPING REQUEST")
    
#     try:
#         # Get parameters
#         data = request.get_json()
        
#         if not data:
#             return jsonify({
#                 "success": False,
#                 "error": "Request body is required"
#             }), 400
        
#         # Validate required fields
#         search_term = data.get('search_term')
#         if not search_term:
#             return jsonify({
#                 "success": False,
#                 "error": "search_term is required"
#             }), 400
        
#         # Extract parameters
#         site_name = data.get('site_name', ['linkedin', 'indeed'])
#         location = data.get('location', 'United States')
#         results_wanted = data.get('results_wanted', 10)
#         is_remote = data.get('is_remote', False)
#         job_type = data.get('job_type', None)
#         hours_old = data.get('hours_old', None)
#         country_indeed = data.get('country_indeed', 'usa')
#         description_format = data.get('description_format', 'plain')
        
#         # Validate
#         if results_wanted > 50:
#             return jsonify({
#                 "success": False,
#                 "error": "Maximum results_wanted is 50 for real-time scraping"
#             }), 400
        
#         logger.info(f"🔴 LIVE SCRAPING: {search_term} in {location}")
#         logger.info(f"   Sites: {site_name}, Results: {results_wanted}")
#         logger.info(f"   ⚠️ NO CACHE - Fetching fresh data from job boards...")
        
#         # 🔴 REAL-TIME SCRAPING - Always fetch full descriptions
#         jobs_df = scrape_jobs(
#             site_name=site_name,
#             search_term=search_term,
#             location=location,
#             results_wanted=results_wanted,
#             is_remote=is_remote,
#             job_type=job_type,
#             hours_old=hours_old,
#             country_indeed=country_indeed,
#             linkedin_fetch_description=True,  # 🔴 ALWAYS fetch full details
#             description_format=description_format,
#             verbose=1
#         )
        
#         scrape_duration = (datetime.now() - scrape_start_time).total_seconds()
        
#         # Check if scraping was successful
#         if jobs_df.empty:
#             logger.warning(f"No jobs found for: {search_term} in {location}")
#             return jsonify({
#                 "success": True,
#                 "count": 0,
#                 "jobs": [],
#                 "message": "No jobs found. Try different search terms.",
#                 "scraping_mode": "REAL_TIME",
#                 "duration_seconds": scrape_duration
#             }), 200
        
#         # Convert to JSON
#         jobs_list = jobs_df.to_dict(orient='records')
        
#         # Clean and enhance data
#         for job in jobs_list:
#             # Clean description
#             if 'description' in job and description_format == 'plain':
#                 job['description_clean'] = clean_description(job['description'])
            
#             # Convert NaN to None
#             for key, value in job.items():
#                 if pd.isna(value):
#                     job[key] = None
            
#             # Add metadata
#             job['data_source'] = 'LIVE_SCRAPING'
#             job['scraped_at'] = scrape_start_time.isoformat()
        
#         # Update metrics
#         total_jobs_scraped += len(jobs_list)
#         last_scrape_time = datetime.now()
        
#         # Calculate data completeness
#         has_description = sum(1 for j in jobs_list if j.get('description'))
#         description_rate = (has_description / len(jobs_list)) * 100 if jobs_list else 0
        
#         logger.info(f"✅ REAL-TIME SCRAPING COMPLETE")
#         logger.info(f"   Jobs found: {len(jobs_list)}")
#         logger.info(f"   Duration: {scrape_duration:.2f}s")
#         logger.info(f"   Description rate: {description_rate:.1f}%")
        
#         return jsonify({
#             "success": True,
#             "count": len(jobs_list),
#             "jobs": jobs_list,
#             "metadata": {
#                 "scraping_mode": "REAL_TIME",
#                 "data_source": "LIVE_JOB_BOARDS",
#                 "cached": False,
#                 "vector_db_used": False,
#                 "scraped_at": scrape_start_time.isoformat(),
#                 "duration_seconds": round(scrape_duration, 2),
#                 "search_params": {
#                     "search_term": search_term,
#                     "location": location,
#                     "sites": site_name
#                 },
#                 "data_quality": {
#                     "jobs_with_description": has_description,
#                     "description_rate": round(description_rate, 1)
#                 }
#             }
#         }), 200
        
#     except Exception as e:
#         logger.error(f"❌ REAL-TIME SCRAPING ERROR: {str(e)}", exc_info=True)
#         return jsonify({
#             "success": False,
#             "error": "Real-time scraping failed. Please try again.",
#             "details": str(e) if app.debug else None,
#             "scraping_mode": "REAL_TIME"
#         }), 500


# @app.route('/scrape', methods=['GET'])
# def scrape_get():
#     """GET endpoint for quick queries"""
#     global total_requests
#     total_requests += 1
    
#     try:
#         site_name = request.args.get('site_name', 'indeed').split(',')
#         search_term = request.args.get('search_term')
#         location = request.args.get('location', 'USA')
#         results_wanted = int(request.args.get('results_wanted', 5))
        
#         if not search_term:
#             return jsonify({
#                 "success": False,
#                 "error": "search_term parameter is required"
#             }), 400
        
#         logger.info(f"🔴 LIVE GET REQUEST: {search_term}")
        
#         # Real-time scraping
#         jobs_df = scrape_jobs(
#             site_name=site_name,
#             search_term=search_term,
#             location=location,
#             results_wanted=results_wanted,
#             linkedin_fetch_description=True,  # 🔴 Always fetch full details
#             verbose=1
#         )
        
#         jobs_list = jobs_df.to_dict(orient='records')
        
#         # Clean NaN values
#         for job in jobs_list:
#             for key, value in job.items():
#                 if pd.isna(value):
#                     job[key] = None
#             job['data_source'] = 'LIVE_SCRAPING'
        
#         return jsonify({
#             "success": True,
#             "count": len(jobs_list),
#             "jobs": jobs_list,
#             "scraping_mode": "REAL_TIME",
#             "note": "Use POST /scrape for more options"
#         }), 200
        
#     except Exception as e:
#         logger.error(f"GET scrape error: {str(e)}")
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500


# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({
#         "success": False,
#         "error": "Endpoint not found",
#         "available_endpoints": ["/", "/health", "/stats", "/scrape"]
#     }), 404


# @app.errorhandler(500)
# def internal_error(error):
#     logger.error(f"Internal server error: {str(error)}")
#     return jsonify({
#         "success": False,
#         "error": "Internal server error"
#     }), 500


# if __name__ == '__main__':
#     print("="*80)
#     print("🔴 JobSpy REAL-TIME API Server")
#     print("="*80)
#     print("Mode: LIVE SCRAPING (No Cache/Vector DB)")
#     print("="*80)
#     print("API running at: http://localhost:5000")
#     print()
#     print("Available endpoints:")
#     print("  GET  /           - API info")
#     print("  GET  /health     - Health check")
#     print("  GET  /stats      - Scraping statistics")
#     print("  GET  /scrape     - Quick scrape (query params)")
#     print("  POST /scrape     - Full scrape (JSON body)")
#     print()
#     print("🔴 REAL-TIME FEATURES:")
#     print("  ✅ Always fetches LIVE data from job boards")
#     print("  ✅ No cached or vector database data")
#     print("  ✅ Full job descriptions fetched in real-time")
#     print("  ✅ Latest job postings")
#     print("  ✅ Complete and accurate information")
#     print()
#     print("⚠️  WARNING:")
#     print("  - Real-time scraping is slower (may take 10-30 seconds)")
#     print("  - May hit rate limits with too many requests")
#     print("  - Use proxies for high-volume scraping")
#     print("="*80)
#     app.run(debug=True, host='0.0.0.0', port=5000)

