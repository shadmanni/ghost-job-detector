from datetime import datetime, timedelta
import hashlib
import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, quote_plus
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer, util

from scraper.models import JobPosting, init_db, get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy global model holder for sentence-transformers
_MODEL_INSTANCE: Optional[SentenceTransformer] = None

# Realistic user agents for rotation
STEALTH_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Desktop screen viewports for rotation
STEALTH_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1366, "height": 768},
    {"width": 1680, "height": 1050},
]


def get_embedding_model() -> SentenceTransformer:
    """Lazy initialization of SentenceTransformer model for similarity matching."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _MODEL_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_INSTANCE


def adaptive_delay(min_s: float = 3.0, max_s: float = 7.0) -> float:
    """Polite, human-like adaptive jitter delay between requests."""
    delay = random.uniform(min_s, max_s)
    logger.debug(f"Adaptive rate-limiting delay: {delay:.2f}s")
    time.sleep(delay)
    return delay


def respect_robots_txt(target_url: str, user_agent: str = "GhostJobBot/1.0") -> bool:
    """Programmatically check robots.txt for a given URL."""
    try:
        parsed = urlparse(target_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rfp = RobotFileParser()
        rfp.set_url(robots_url)
        rfp.read()
        can_fetch = rfp.can_fetch(user_agent, target_url)
        if not can_fetch:
            logger.warning(f"Robots.txt disallows scraping path: {target_url}")
        return can_fetch
    except Exception as e:
        logger.warning(f"Failed to fetch/parse robots.txt for {target_url}: {e}. Proceeding with caution.")
        return True


def parse_relative_time(time_str: str) -> datetime:
    """Parse relative time strings like '3 days ago', '1 week ago', 'just posted' into a datetime."""
    now = datetime.utcnow()
    if not time_str:
        return now

    s = time_str.strip().lower()
    try:
        # ISO date matching
        if re.match(r"^\d{4}-\d{2}-\d{2}", s):
            return datetime.fromisoformat(s[:10])

        if "just" in s or "today" in s or "hour" in s or "minute" in s:
            return now

        days_match = re.search(r"(\d+)\s+day", s)
        if days_match:
            return now - timedelta(days=int(days_match.group(1)))

        weeks_match = re.search(r"(\d+)\s+week", s)
        if weeks_match:
            return now - timedelta(weeks=int(weeks_match.group(1)))

        months_match = re.search(r"(\d+)\s+month", s)
        if months_match:
            return now - timedelta(days=int(months_match.group(1)) * 30)
    except Exception:
        pass

    return now


def career_page_scraper(
    url: str,
    company: str = None,
    html_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Given a company careers page URL, use stealth Playwright (or html_override) to render
    and parse job listings with BeautifulSoup.
    """
    logger.info(f"Scraping career page for company: {company or 'Unknown'} at URL: {url}")
    postings: List[Dict[str, Any]] = []

    if html_override:
        html = html_override
    else:
        if not respect_robots_txt(url):
            logger.warning(f"Robots.txt restrictions apply to career page {url}. Proceeding cautiously.")

        adaptive_delay(2.0, 4.0)

        try:
            from playwright.sync_api import sync_playwright

            ua = random.choice(STEALTH_USER_AGENTS)
            vp = random.choice(STEALTH_VIEWPORTS)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=ua,
                    viewport=vp,
                    locale="en-US",
                    timezone_id="America/New_York",
                )
                # Inject stealth evasions
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    window.chrome = { runtime: {} };
                """)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

                # Human-like incremental scrolling
                for _ in range(random.randint(2, 4)):
                    page.mouse.wheel(0, random.randint(300, 600))
                    time.sleep(random.uniform(0.4, 0.9))

                time.sleep(1.5)
                html = page.content()
                browser.close()
        except Exception as e:
            logger.error(f"Playwright error while rendering career page {url}: {e}")
            return postings

    soup = BeautifulSoup(html, "html.parser")
    domain_company = company or urlparse(url).netloc.split(".")[0].capitalize()

    job_elements = soup.find_all(
        lambda tag: tag.name in ["div", "li", "tr", "article", "a"]
        and any(
            k in str(tag.get("class", [])).lower()
            or k in str(tag.get("id", "")).lower()
            or k in (tag.get("href") or "").lower()
            for k in ["job", "career", "posting", "position", "opening", "opportunity"]
        )
    )

    seen_urls = set()

    for el in job_elements:
        title_el = el.find(["h1", "h2", "h3", "h4", "h5", "a", "span", "strong"])
        if not title_el and el.name == "a":
            title_el = el
        title = title_el.get_text(strip=True) if title_el else ""

        if not title or len(title) < 3 or len(title) > 150:
            continue

        link_tag = el if el.name == "a" else el.find("a")
        href = link_tag.get("href") if link_tag else None
        if not href or href.startswith("#") or href.startswith("javascript:"):
            job_url = url
        else:
            job_url = urljoin(url, href)

        if job_url in seen_urls:
            continue
        seen_urls.add(job_url)

        description = el.get_text(separator=" ", strip=True)
        if len(description) < 20:
            description = f"Job listing for {title} at {domain_company}. Full details and application via official portal."

        # Extract salary if present
        salary = None
        for text_piece in el.stripped_strings:
            if any(curr in text_piece for curr in ["$", "€", "£", "EUR", "GBP", "USD"]) or "salary" in text_piece.lower():
                salary = text_piece
                break

        # Extract location if present
        location = None
        loc_el = el.find(attrs={"class": lambda c: c and any(w in str(c).lower() for w in ["loc", "city", "place"])})
        if loc_el:
            location = loc_el.get_text(strip=True)

        # Extract date
        posted_date = datetime.utcnow()
        time_el = el.find("time")
        if time_el:
            date_val = time_el.get("datetime") or time_el.get_text(strip=True)
            posted_date = parse_relative_time(date_val)

        postings.append({
            "company": domain_company,
            "title": title,
            "description": description,
            "url": job_url,
            "source": "Career Page",
            "posted_date": posted_date,
            "salary_listed": salary,
            "location": location,
        })

    logger.info(f"Career page scraper extracted {len(postings)} listings from {url}")
    return postings


def indeed_scraper(
    query: str,
    location: str,
    company: Optional[str] = None,
    html_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scrape public Indeed search results respecting robots.txt with adaptive rate limiting and stealth."""
    base_url = "https://www.indeed.com"
    search_q = f"{company} {query}".strip() if company else query
    target_url = f"{base_url}/jobs?q={quote_plus(search_q)}&l={quote_plus(location)}"

    postings: List[Dict[str, Any]] = []

    if not html_override:
        if not respect_robots_txt(target_url):
            logger.warning(f"Indeed robots.txt forbids scraping {target_url}. Skipping.")
            return postings

        adaptive_delay(3.0, 6.0)

        try:
            from playwright.sync_api import sync_playwright

            ua = random.choice(STEALTH_USER_AGENTS)
            vp = random.choice(STEALTH_VIEWPORTS)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=ua,
                    viewport=vp,
                    locale="en-US",
                )
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)
                page = context.new_page()
                page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                time.sleep(2)
                html = page.content()
                browser.close()
        except Exception as e:
            logger.error(f"Playwright error during Indeed scraping: {e}")
            return postings
    else:
        html = html_override

    soup = BeautifulSoup(html, "html.parser")

    job_cards = soup.select(".job_seen_beacon, .result, div[data-jk]")
    if not job_cards:
        job_cards = soup.find_all("div", class_=lambda c: c and "job" in c.lower())

    for card in job_cards:
        title_el = card.select_one("h2.jobTitle, a.jcs-JobTitle, .jobTitle span")
        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            continue

        company_el = card.select_one("[data-testid='company-name'], .companyName, .company")
        company_name = company or (company_el.get_text(strip=True) if company_el else "Unknown")

        link_el = card.select_one("a[data-jk], a.jcs-JobTitle, h2.jobTitle a")
        jk_val = card.get("data-jk") or (link_el.get("data-jk") if link_el else None)
        if jk_val:
            job_url = f"https://www.indeed.com/viewjob?jk={jk_val}"
        elif link_el and link_el.get("href"):
            job_url = urljoin(base_url, link_el.get("href"))
        else:
            job_url = target_url

        snippet_el = card.select_one(".job-snippet, .underline, ul")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else f"Indeed listing for {title} at {company_name}"

        salary_el = card.select_one(".salary-snippet-container, .metadata.salary-snippet-container, .attribute_snippet")
        salary = salary_el.get_text(strip=True) if salary_el else None

        loc_el = card.select_one("[data-testid='text-location'], .companyLocation")
        loc_val = loc_el.get_text(strip=True) if loc_el else location

        postings.append({
            "company": company_name,
            "title": title,
            "description": snippet,
            "url": job_url,
            "source": "Indeed",
            "posted_date": datetime.utcnow(),
            "salary_listed": salary,
            "location": loc_val,
        })

    logger.info(f"Indeed scraper extracted {len(postings)} listings for '{query}' in '{location}'")
    return postings


def linkedin_scraper(
    query: str,
    location: str = "United States",
    company: Optional[str] = None,
    html_override: Optional[str] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Scrape public LinkedIn job search results (guest view, no authentication required).
    Uses Playwright with stealth configurations, realistic user-agent rotation, and human delays.
    """
    search_keywords = f"{company} {query}".strip() if company else query
    target_url = f"https://www.linkedin.com/jobs/search?keywords={quote_plus(search_keywords)}&location={quote_plus(location)}"
    logger.info(f"Scraping LinkedIn Public Search: '{search_keywords}' in '{location}' ({target_url})")

    postings: List[Dict[str, Any]] = []

    if html_override:
        html = html_override
    else:
        if not respect_robots_txt(target_url):
            logger.warning(f"LinkedIn robots.txt advises caution on {target_url}.")

        adaptive_delay(3.0, 6.0)

        try:
            from playwright.sync_api import sync_playwright

            ua = random.choice(STEALTH_USER_AGENTS)
            vp = random.choice(STEALTH_VIEWPORTS)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=ua,
                    viewport=vp,
                    locale="en-US",
                )
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    window.chrome = { runtime: {} };
                """)
                page = context.new_page()
                page.goto(target_url, wait_until="domcontentloaded", timeout=30000)

                # Realistic incremental scroll down to trigger dynamic card loading
                for _ in range(random.randint(2, 4)):
                    page.mouse.wheel(0, random.randint(400, 800))
                    time.sleep(random.uniform(0.6, 1.2))

                time.sleep(2)
                html = page.content()
                browser.close()
        except Exception as e:
            logger.warning(f"Playwright unavailable or error ({e}). Attempting direct LinkedIn guest HTTP search...")
            try:
                import requests
                guest_api_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={quote_plus(search_keywords)}&location={quote_plus(location)}&start=0"
                headers = {
                    "User-Agent": random.choice(STEALTH_USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                resp = requests.get(guest_api_url, headers=headers, timeout=12)
                if resp.status_code == 200 and len(resp.text) > 100:
                    html = resp.text
                else:
                    resp2 = requests.get(target_url, headers=headers, timeout=12)
                    html = resp2.text if resp2.status_code == 200 else ""
            except Exception as e2:
                logger.error(f"LinkedIn HTTP fallback error: {e2}")
                return postings

        if not html:
            logger.warning(f"No HTML obtained for LinkedIn search: {search_keywords}")
            return postings

    soup = BeautifulSoup(html, "html.parser")

    # LinkedIn public cards selectors
    cards = soup.select(".base-card, .job-search-card, li:has(.base-card)")
    if not cards:
        cards = soup.find_all("div", class_=lambda c: c and "job-card" in c.lower())

    seen_urls = set()

    for card in cards[:limit]:
        title_el = card.select_one(".base-search-card__title, h3.base-search-card__title, .job-card-list__title")
        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            continue

        company_el = card.select_one(".base-search-card__subtitle, h4.base-search-card__subtitle a, .job-card-container__company-name")
        card_company = company or (company_el.get_text(strip=True) if company_el else "Unknown")

        link_el = card.select_one("a.base-card__full-link, a[data-tracking-control-name], a.job-card-list__title")
        raw_url = link_el.get("href") if link_el else None
        if raw_url:
            # Strip tracking query params
            job_url = raw_url.split("?")[0]
        else:
            job_url = target_url

        if job_url in seen_urls:
            continue
        seen_urls.add(job_url)

        # Location
        loc_el = card.select_one(".job-search-card__location, .job-card-container__metadata-item")
        loc_val = loc_el.get_text(strip=True) if loc_el else location

        # Date posted
        date_el = card.select_one("time.job-search-card__listdate, time")
        if date_el:
            dt_str = date_el.get("datetime") or date_el.get_text(strip=True)
            posted_date = parse_relative_time(dt_str)
        else:
            posted_date = datetime.utcnow()

        # Salary snippet
        salary_el = card.select_one(".job-search-card__salary-info, .salary-snippet")
        salary_val = salary_el.get_text(strip=True) if salary_el else None

        # Description / Snippet
        snippet_el = card.select_one(".job-search-card__snippet, p")
        description = snippet_el.get_text(" ", strip=True) if snippet_el else f"Public LinkedIn requisition for {title} at {card_company} in {loc_val}."

        postings.append({
            "company": card_company,
            "title": title,
            "description": description,
            "url": job_url,
            "source": "LinkedIn",
            "posted_date": posted_date,
            "salary_listed": salary_val,
            "location": loc_val,
        })

    logger.info(f"LinkedIn scraper extracted {len(postings)} public postings for '{search_keywords}'")
    return postings


def save_postings(
    postings: List[Dict[str, Any]],
    db_path: str = "data/ghostjobs.db",
    similarity_threshold: float = 0.92,
) -> Dict[str, int]:
    """Upsert postings into JobPosting table with content hashing & sentence-transformers cosine similarity deduplication."""
    if not postings:
        return {"total": 0, "new": 0, "reposts": 0, "skipped": 0}

    init_db(db_path)
    session: Session = get_session(db_path)
    model = get_embedding_model()

    new_count = 0
    repost_count = 0
    skipped_count = 0

    try:
        for p_data in postings:
            company_name = p_data.get("company", "Unknown")
            posting_url = p_data.get("url")
            title = p_data.get("title", "Untitled Position")
            new_desc = p_data.get("description", "")
            location = p_data.get("location")
            salary_listed = p_data.get("salary_listed")
            posted_date = p_data.get("posted_date", datetime.utcnow())

            # 1. Content hash computation for quick exact deduplication
            hash_input = f"{company_name.lower().strip()}:{title.lower().strip()}:{new_desc[:250].strip()}"
            content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

            # Check exact URL duplicate
            if posting_url:
                existing_by_url = session.query(JobPosting).filter_by(url=posting_url).first()
                if existing_by_url:
                    skipped_count += 1
                    continue

            # Query recent postings from same company to detect reposts
            recent_postings = (
                session.query(JobPosting)
                .filter(JobPosting.company == company_name)
                .order_by(JobPosting.scraped_at.desc())
                .limit(20)
                .all()
            )

            repost_of_id = None

            if recent_postings and new_desc and len(new_desc.strip()) > 10:
                recent_descs = [p.description or "" for p in recent_postings]
                # Compute embeddings
                new_emb = model.encode(new_desc, convert_to_tensor=True)
                recent_embs = model.encode(recent_descs, convert_to_tensor=True)

                # Compute cosine similarities
                sim_matrix = util.cos_sim(new_emb, recent_embs)[0]
                max_sim_idx = int(sim_matrix.argmax().item())
                max_sim_val = float(sim_matrix[max_sim_idx].item())

                logger.debug(
                    f"Comparing description for '{title}' at '{company_name}'. Max similarity: {max_sim_val:.4f}"
                )

                if max_sim_val >= similarity_threshold:
                    matched_posting = recent_postings[max_sim_idx]
                    repost_of_id = matched_posting.id
                    logger.info(
                        f"Identified repost of JobPosting ID {repost_of_id} (similarity {max_sim_val:.4f} >= {similarity_threshold})"
                    )
                    repost_count += 1
                else:
                    new_count += 1
            else:
                new_count += 1

            new_job = JobPosting(
                company=company_name,
                title=title,
                description=new_desc,
                url=posting_url,
                source=p_data.get("source", "Unknown"),
                posted_date=posted_date,
                scraped_at=datetime.utcnow(),
                salary_listed=salary_listed,
                location=location,
                content_hash=content_hash,
                repost_of_id=repost_of_id,
            )
            session.add(new_job)

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving postings to DB: {e}")
        raise e
    finally:
        session.close()

    summary = {
        "total": len(postings),
        "new": new_count,
        "reposts": repost_count,
        "skipped": skipped_count,
    }
    logger.info(f"Posting save operation completed: {summary}")
    return summary
