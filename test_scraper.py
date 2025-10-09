from jobspy import scrape_jobs
import pandas as pd

print("Testing JobSpy - Job Scraper\n")
print("=" * 50)

# Test scraping jobs
jobs = scrape_jobs(
    site_name=["indeed", "linkedin"],
    search_term="python developer",
    location="United States",
    results_wanted=10,
    hours_old=72,
    country_indeed="usa",
    verbose=2
)

print(f"\n✅ Successfully scraped {len(jobs)} jobs!")
print("\n" + "=" * 50)
print("Sample Job Data:")
print("=" * 50)

if len(jobs) > 0:
    print(jobs[['title', 'company', 'location', 'site']].head())
    jobs.to_csv("jobs_output.csv", index=False)
    print(f"\n💾 Saved {len(jobs)} jobs to 'jobs_output.csv'")
else:
    print("No jobs found. Try different search terms.")