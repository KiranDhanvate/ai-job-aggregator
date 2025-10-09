from jobspy import scrape_jobs
import pandas as pd

print("Scraping jobs with FULL descriptions...\n")
print("="*80)

# Scrape with full descriptions enabled
jobs = scrape_jobs(
    site_name=["indeed", "linkedin"],
    search_term="python developer",
    location="United States",
    results_wanted=5,
    
    # ⭐ Enable full descriptions
    linkedin_fetch_description=True,
    description_format="plain",  # Options: "plain", "markdown", "html"
    
    hours_old=72,
    country_indeed="usa",
    verbose=2
)

print(f"\n✅ Successfully scraped {len(jobs)} jobs!\n")

# Save to CSV
jobs.to_csv("jobs_full_descriptions.csv", index=False)
print("💾 Saved to 'jobs_full_descriptions.csv'\n")

# Display each job with its full description
for idx, job in jobs.iterrows():
    print("="*80)
    print(f"JOB #{idx+1}")
    print("="*80)
    print(f"Title:    {job['title']}")
    print(f"Company:  {job['company']}")
    print(f"Location: {job['location']}")
    print(f"Site:     {job['site']}")
    print(f"Posted:   {job['date_posted']}")
    print(f"\nURL: {job['job_url']}")
    print("\n" + "-"*80)
    print("DESCRIPTION:")
    print("-"*80)
    
    # Print full description (first 1000 characters)
    desc = job['description']
    if pd.notna(desc):
        print(desc[:1000])
        if len(desc) > 1000:
            print(f"\n... (truncated, total length: {len(desc)} characters)")
    else:
        print("No description available")
    
    print("\n")

print("="*80)
print("✅ Done! Check 'jobs_full_descriptions.csv' for complete data")
print("="*80)