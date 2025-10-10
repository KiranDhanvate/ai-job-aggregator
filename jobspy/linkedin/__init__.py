# jobspy/linkedin/__init__.py - Enhanced Version

from __future__ import annotations

import math
import random
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, urlunparse, unquote
from functools import wraps

import regex as re
from bs4 import BeautifulSoup
from bs4.element import Tag

from jobspy.exception import LinkedInException
from jobspy.linkedin.constant import headers
from jobspy.linkedin.util import (
    is_job_remote,
    job_type_code,
    parse_job_type,
    parse_job_level,
    parse_company_industry
)
from jobspy.model import (
    JobPost,
    Location,
    JobResponse,
    Country,
    Compensation,
    DescriptionFormat,
    Scraper,
    ScraperInput,
    Site,
)
from jobspy.util import (
    extract_emails_from_text,
    currency_parser,
    markdown_converter,
    plain_converter,
    create_session,
    remove_attributes,
    create_logger,
)

log = create_logger("LinkedIn")


def retry_with_backoff(max_retries=3, base_delay=2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        log.error(f"Failed after {max_retries} attempts: {str(e)}")
                        return None
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"Retry {attempt + 1}/{max_retries} after {delay}s: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator


class LinkedIn(Scraper):
    base_url = "https://www.linkedin.com"
    delay = 3
    band_delay = 4
    jobs_per_page = 25

    def __init__(
        self, proxies: list[str] | str | None = None, ca_cert: str | None = None, user_agent: str | None = None
    ):
        """
        Initializes LinkedInScraper with the LinkedIn job search url
        """
        super().__init__(Site.LINKEDIN, proxies=proxies, ca_cert=ca_cert)
        self.session = create_session(
            proxies=self.proxies,
            ca_cert=ca_cert,
            is_tls=False,
            has_retry=True,
            delay=5,
            clear_cookies=True,
        )
        self.session.headers.update(headers)
        self.scraper_input = None
        self.country = "worldwide"
        self.job_url_direct_regex = re.compile(r'(?<=\?url=)[^"]+')
        
        # Adaptive rate limiting
        self.request_count = 0
        self.rate_limit_hit = False
        self.adaptive_delay = 3
        self.failed_details_count = 0
        
        # Proxy rotation
        self.proxy_list = proxies if isinstance(proxies, list) else [proxies] if proxies else []
        self.current_proxy_index = 0

    def _adaptive_sleep(self):
        """Increase delay if we're getting close to rate limits"""
        if self.request_count > 30:
            self.adaptive_delay = min(10, self.adaptive_delay * 1.2)
        elif self.failed_details_count > 3:
            self.adaptive_delay = min(15, self.adaptive_delay * 1.5)
            
        sleep_time = random.uniform(self.adaptive_delay, self.adaptive_delay + 2)
        log.debug(f"Sleeping for {sleep_time:.2f}s (adaptive delay: {self.adaptive_delay:.2f}s)")
        time.sleep(sleep_time)
        
    def _rotate_proxy(self):
        """Rotate to next proxy"""
        if self.proxy_list and len(self.proxy_list) > 1:
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
            proxy = self.proxy_list[self.current_proxy_index]
            self.session.proxies.update({"http": proxy, "https": proxy})
            log.info(f"Rotated to proxy {self.current_proxy_index + 1}/{len(self.proxy_list)}")

    def scrape(self, scraper_input: ScraperInput) -> JobResponse:
        """
        Scrapes LinkedIn for jobs with scraper_input criteria
        :param scraper_input:
        :return: job_response
        """
        self.scraper_input = scraper_input
        job_list: list[JobPost] = []
        seen_ids = set()
        start = scraper_input.offset // 10 * 10 if scraper_input.offset else 0
        seconds_old = (
            scraper_input.hours_old * 3600 if scraper_input.hours_old else None
        )
        continue_search = (
            lambda: len(job_list) < scraper_input.results_wanted and start < 1000
        )
        
        while continue_search():
            self.request_count += 1
            log.info(
                f"search page: {self.request_count} / {math.ceil(scraper_input.results_wanted / 10)}"
            )
            
            # Rotate proxy every 10 requests
            if self.request_count % 10 == 0:
                self._rotate_proxy()
            
            params = {
                "keywords": scraper_input.search_term,
                "location": scraper_input.location,
                "distance": scraper_input.distance,
                "f_WT": 2 if scraper_input.is_remote else None,
                "f_JT": (
                    job_type_code(scraper_input.job_type)
                    if scraper_input.job_type
                    else None
                ),
                "pageNum": 0,
                "start": start,
                "f_AL": "true" if scraper_input.easy_apply else None,
                "f_C": (
                    ",".join(map(str, scraper_input.linkedin_company_ids))
                    if scraper_input.linkedin_company_ids
                    else None
                ),
            }
            if seconds_old is not None:
                params["f_TPR"] = f"r{seconds_old}"

            params = {k: v for k, v in params.items() if v is not None}
            
            try:
                response = self.session.get(
                    f"{self.base_url}/jobs-guest/jobs/api/seeMoreJobPostings/search?",
                    params=params,
                    timeout=10,
                )
                if response.status_code not in range(200, 400):
                    if response.status_code == 429:
                        err = f"429 Response - Rate limited. Increasing delay."
                        self.rate_limit_hit = True
                        self.adaptive_delay = min(20, self.adaptive_delay * 2)
                        log.warning(err)
                        time.sleep(self.adaptive_delay)
                        continue
                    else:
                        err = f"LinkedIn response status code {response.status_code}"
                        err += f" - {response.text[:200]}"
                    log.error(err)
                    return JobResponse(jobs=job_list)
            except Exception as e:
                if "Proxy responded with" in str(e):
                    log.error(f"LinkedIn: Bad proxy, rotating...")
                    self._rotate_proxy()
                else:
                    log.error(f"LinkedIn: {str(e)}")
                return JobResponse(jobs=job_list)

            soup = BeautifulSoup(response.text, "html.parser")
            job_cards = soup.find_all("div", class_="base-search-card")
            
            if len(job_cards) == 0:
                log.warning("No job cards found, ending search")
                return JobResponse(jobs=job_list)

            for job_card in job_cards:
                href_tag = job_card.find("a", class_="base-card__full-link")
                if href_tag and "href" in href_tag.attrs:
                    href = href_tag.attrs["href"].split("?")[0]
                    job_id = href.split("-")[-1]

                    if job_id in seen_ids:
                        continue
                    seen_ids.add(job_id)

                    try:
                        fetch_desc = scraper_input.linkedin_fetch_description
                        job_post = self._process_job(job_card, job_id, fetch_desc)
                        if job_post:
                            job_list.append(job_post)
                            log.debug(f"Added job: {job_post.title} at {job_post.company_name}")
                        if not continue_search():
                            break
                    except Exception as e:
                        log.error(f"Failed to process job {job_id}: {str(e)}", exc_info=True)
                        # Continue processing other jobs instead of crashing
                        continue

            if continue_search():
                self._adaptive_sleep()
                start += len(job_cards)

        job_list = job_list[: scraper_input.results_wanted]
        log.info(f"Scraping completed. Collected {len(job_list)} jobs")
        return JobResponse(jobs=job_list)

    def _process_job(
        self, job_card: Tag, job_id: str, full_descr: bool
    ) -> Optional[JobPost]:
        """Process job card with enhanced error handling"""
        try:
            salary_tag = job_card.find("span", class_="job-search-card__salary-info")

            compensation = description = None
            if salary_tag:
                salary_text = salary_tag.get_text(separator=" ").strip()
                salary_values = [currency_parser(value) for value in salary_text.split("-")]
                if len(salary_values) >= 2:
                    salary_min = salary_values[0]
                    salary_max = salary_values[1]
                    currency = salary_text[0] if salary_text[0] != "$" else "USD"

                    compensation = Compensation(
                        min_amount=int(salary_min),
                        max_amount=int(salary_max),
                        currency=currency,
                    )

            title_tag = job_card.find("span", class_="sr-only")
            title = title_tag.get_text(strip=True) if title_tag else "N/A"

            company_tag = job_card.find("h4", class_="base-search-card__subtitle")
            company_a_tag = company_tag.find("a") if company_tag else None
            company_url = (
                urlunparse(urlparse(company_a_tag.get("href"))._replace(query=""))
                if company_a_tag and company_a_tag.has_attr("href")
                else ""
            )
            company = company_a_tag.get_text(strip=True) if company_a_tag else "N/A"

            metadata_card = job_card.find("div", class_="base-search-card__metadata")
            location = self._get_location(metadata_card)

            datetime_tag = (
                metadata_card.find("time", class_="job-search-card__listdate")
                if metadata_card
                else None
            )
            date_posted = None
            if datetime_tag and "datetime" in datetime_tag.attrs:
                datetime_str = datetime_tag["datetime"]
                try:
                    date_posted = datetime.strptime(datetime_str, "%Y-%m-%d")
                except:
                    date_posted = None
                    
            job_details = {}
            if full_descr:
                job_details = self._get_job_details(job_id)
                description = job_details.get("description")
                
            is_remote = is_job_remote(title, description, location)

            return JobPost(
                id=f"li-{job_id}",
                title=title,
                company_name=company,
                company_url=company_url,
                location=location,
                is_remote=is_remote,
                date_posted=date_posted,
                job_url=f"{self.base_url}/jobs/view/{job_id}",
                compensation=compensation,
                job_type=job_details.get("job_type"),
                job_level=job_details.get("job_level", "").lower() if job_details.get("job_level") else None,
                company_industry=job_details.get("company_industry"),
                description=job_details.get("description"),
                job_url_direct=job_details.get("job_url_direct"),
                emails=extract_emails_from_text(description) if description else None,
                company_logo=job_details.get("company_logo"),
                job_function=job_details.get("job_function"),
            )
        except Exception as e:
            log.error(f"Error processing job card for {job_id}: {str(e)}")
            log.debug(f"Job card HTML: {job_card.prettify()[:500]}")
            return None

    @retry_with_backoff(max_retries=3, base_delay=2)
    def _get_job_details(self, job_id: str) -> dict:
        """
        Retrieves job description and other job details with retry logic
        :param job_id: LinkedIn job ID
        :return: dict with job details
        """
        try:
            response = self.session.get(
                f"{self.base_url}/jobs/view/{job_id}", timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            self.failed_details_count += 1
            log.warning(f"Failed to fetch details for job {job_id}: {str(e)}")
            return {}
            
        if "linkedin.com/signup" in response.url:
            log.warning(f"Redirected to signup for job {job_id}")
            self.failed_details_count += 1
            return {}

        soup = BeautifulSoup(response.text, "html.parser")
        div_content = soup.find(
            "div", class_=lambda x: x and "show-more-less-html__markup" in x
        )
        description = None
        if div_content is not None:
            div_content = remove_attributes(div_content)
            description = div_content.prettify(formatter="html")
            if self.scraper_input.description_format == DescriptionFormat.MARKDOWN:
                description = markdown_converter(description)
            elif self.scraper_input.description_format == DescriptionFormat.PLAIN:
                description = plain_converter(description)
                
        h3_tag = soup.find(
            "h3", text=lambda text: text and "Job function" in text.strip()
        )

        job_function = None
        if h3_tag:
            job_function_span = h3_tag.find_next(
                "span", class_="description__job-criteria-text"
            )
            if job_function_span:
                job_function = job_function_span.get_text(strip=True)

        company_logo = (
            logo_image.get("data-delayed-url")
            if (logo_image := soup.find("img", {"class": "artdeco-entity-image"}))
            else None
        )
        
        # Reset failed count on success
        if description:
            self.failed_details_count = max(0, self.failed_details_count - 1)
            
        return {
            "description": description,
            "job_level": parse_job_level(soup),
            "company_industry": parse_company_industry(soup),
            "job_type": parse_job_type(soup),
            "job_url_direct": self._parse_job_url_direct(soup),
            "company_logo": company_logo,
            "job_function": job_function,
        }

    def _get_location(self, metadata_card: Optional[Tag]) -> Location:
        """
        Extracts the location data from the job metadata card.
        :param metadata_card
        :return: location
        """
        location = Location(country=Country.from_string(self.country))
        if metadata_card is not None:
            location_tag = metadata_card.find(
                "span", class_="job-search-card__location"
            )
            location_string = location_tag.text.strip() if location_tag else "N/A"
            parts = location_string.split(", ")
            if len(parts) == 2:
                city, state = parts
                location = Location(
                    city=city,
                    state=state,
                    country=Country.from_string(self.country),
                )
            elif len(parts) == 3:
                city, state, country = parts
                country = Country.from_string(country)
                location = Location(city=city, state=state, country=country)
        return location

    def _parse_job_url_direct(self, soup: BeautifulSoup) -> str | None:
        """
        Gets the job url direct from job page
        :param soup:
        :return: str
        """
        job_url_direct = None
        job_url_direct_content = soup.find("code", id="applyUrl")
        if job_url_direct_content:
            job_url_direct_match = self.job_url_direct_regex.search(
                job_url_direct_content.decode_contents().strip()
            )
            if job_url_direct_match:
                job_url_direct = unquote(job_url_direct_match.group())

        return job_url_direct