from datetime import datetime
import logging
import os
import random
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer, util

from scraper.models import JobPosting, init_db, get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy global model holder for sentence-transformers
_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy initialization of SentenceTransformer model for similarity matching."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        _MODEL_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL_INSTANCE


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


def career_page_scraper(
    url: str,
    company: str = None,
    html_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Given a company careers page URL, use Playwright (or html_override) to render

    and parse job listings with BeautifulSoup.
    """
    logger.info(f"Scraping career page for company: {company or 'Unknown'} at URL: {url}")
    postings: List[Dict[str, Any]] = []

    if html_override:
        html = html_override
    else:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Scroll down slightly to trigger dynamic lazy loading if present
                page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
                time.sleep(1)
                html = page.content()
                browser.close()
        except Exception as e:
            logger.error(f"Playwright error while rendering career page {url}: {e}")
            return postings

    soup = BeautifulSoup(html, "html.parser")
    domain_company = company or urlparse(url).netloc.split(".")[0].capitalize()

    # Find potential job card / link elements
    # Common job board container patterns (Lever, Greenhouse, Workday, Ashby, general career lists)
    job_elements = soup.find_all(
        lambda tag: tag.name in ["div", "li", "tr", "article", "a"]
        and any(
            k in str(tag.get("class", [])).lower() or k in str(tag.get("id", "")).lower() or k in (tag.get("href") or "").lower()
            for k in ["job", "career", "posting", "position", "opening", "opportunity"]
        )
    )

    seen_urls = set()

    for el in job_elements:
        # Extract title
        title_el = el.find(["h1", "h2", "h3", "h4", "h5", "a", "span", "strong"])
        if not title_el and el.name == "a":
            title_el = el
        title = title_el.get_text(strip=True) if title_el else ""

        if not title or len(title) < 3 or len(title) > 150:
            continue

        # Extract URL
        link_tag = el if el.name == "a" else el.find("a")
        href = link_tag.get("href") if link_tag else None
        if not href or href.startswith("#") or href.startswith("javascript:"):
            job_url = url
        else:
            job_url = urljoin(url, href)

        if job_url in seen_urls:
            continue
        seen_urls.add(job_url)

        # Extract description / snippet text
        description = el.get_text(separator=" ", strip=True)
        if len(description) < 20:
            description = f"Job listing for {title} at {domain_company}. Apply online."

        # Extract salary if present
        salary = None
        for text_piece in el.stripped_strings:
            if "$" in text_piece or "EUR" in text_piece or "GBP" in text_piece or "salary" in text_piece.lower():
                salary = text_piece
                break

        postings.append({
            "company": domain_company,
            "title": title,
            "description": description,
            "url": job_url,
            "source": "Career Page",
            "posted_date": datetime.utcnow(),
            "salary_listed": salary,
        })

    logger.info(f"Career page scraper extracted {len(postings)} listings from {url}")
    return postings


def indeed_scraper(
    query: str,
    location: str,
    company: Optional[str] = None,
    html_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scrape public Indeed search results respecting robots.txt and adding polite rate limiting."""
    base_url = "https://www.indeed.com"
    target_url = f"{base_url}/jobs?q={query.replace(' ', '+')}&l={location.replace(' ', '+')}"
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

    postings: List[Dict[str, Any]] = []

    if not html_override:
        if not respect_robots_txt(target_url, user_agent):
            logger.warning(f"Indeed robots.txt forbids scraping {target_url}. Skipping.")
            return postings

        # Polite rate limiting delay (2-5s)
        delay = random.uniform(2.0, 5.0)
        logger.info(f"Applying polite rate-limiting delay of {delay:.2f}s before request...")
        time.sleep(delay)

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=user_agent)
                page = context.new_page()
                page.goto(target_url, wait_until="networkidle", timeout=30000)
                html = page.content()
                browser.close()
        except Exception as e:
            logger.error(f"Playwright error during Indeed scraping: {e}")
            return postings
    else:
        html = html_override

    soup = BeautifulSoup(html, "html.parser")

    # Indeed job cards parsing
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

        postings.append({
            "company": company_name,
            "title": title,
            "description": snippet,
            "url": job_url,
            "source": "Indeed",
            "posted_date": datetime.utcnow(),
            "salary_listed": salary,
        })

    logger.info(f"Indeed scraper extracted {len(postings)} listings for '{query}' in '{location}'")
    return postings


def save_postings(
    postings: List[Dict[str, Any]],
    db_path: str = "data/ghostjobs.db",
    similarity_threshold: float = 0.92,
) -> Dict[str, int]:
    """Upsert postings into JobPosting table with sentence-transformers cosine similarity deduplication."""
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

            # Check exact URL duplicate first
            if posting_url:
                existing_by_url = session.query(JobPosting).filter_by(url=posting_url).first()
                if existing_by_url:
                    skipped_count += 1
                    continue

            # Query last 20 postings from SAME company
            recent_postings = (
                session.query(JobPosting)
                .filter(JobPosting.company == company_name)
                .order_by(JobPosting.scraped_at.desc())
                .limit(20)
                .all()
            )

            new_desc = p_data.get("description", "")
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

                logger.debug(f"Comparing description for '{p_data.get('title')}' against company '{company_name}' previous postings. Max similarity: {max_sim_val:.4f}")

                if max_sim_val >= similarity_threshold:
                    matched_posting = recent_postings[max_sim_idx]
                    repost_of_id = matched_posting.id
                    logger.info(f"Identified repost of JobPosting ID {repost_of_id} (similarity {max_sim_val:.4f} >= {similarity_threshold})")
                    repost_count += 1
                else:
                    new_count += 1
            else:
                new_count += 1

            new_job = JobPosting(
                company=company_name,
                title=p_data.get("title", "Untitled Position"),
                description=new_desc,
                url=posting_url,
                source=p_data.get("source", "Unknown"),
                posted_date=p_data.get("posted_date", datetime.utcnow()),
                scraped_at=datetime.utcnow(),
                salary_listed=p_data.get("salary_listed"),
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
