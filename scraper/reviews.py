from datetime import datetime
import json
import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from scraper.models import CompanyReview, CompanySentiment, init_db, get_session
from scraper.jobboards import STEALTH_USER_AGENTS, adaptive_delay

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_praw_client():
    """Initialize PRAW Reddit client using environment credentials if available."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "GhostJobDetector/1.0")

    if not client_id or not client_secret or "your_" in client_id.lower():
        logger.info(
            "Reddit API credentials (REDDIT_CLIENT_ID) not configured or are placeholders. "
            "Using Zero-API Public JSON & Web Scraping fallback."
        )
        return None

    try:
        import praw

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_async=False,
        )
        return reddit
    except Exception as e:
        logger.error(f"Failed to initialize PRAW Reddit client: {e}")
        return None


def reddit_public_json_scraper(
    company: str,
    limit: int = 25,
    subreddits: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Method A: Safely fetch public Reddit thread listings and comments using public URL endpoints

    with .json extensions, polite browser headers, and exponential backoff retry.
    Zero developer API credentials required.
    """
    if subreddits is None:
        subreddits = ["recruitinghell", "jobs", "antiwork"]

    reviews: List[Dict[str, Any]] = []
    seen_texts = set()

    # Search queries to target
    search_targets = []
    for sub in subreddits:
        search_targets.append(
            f"https://www.reddit.com/r/{sub}/search.json?q={quote_plus(company)}&restrict_sr=1&sort=new&limit={limit}"
        )
    # General global search
    search_targets.append(
        f"https://www.reddit.com/search.json?q={quote_plus(company)}+ghosted+OR+interview+OR+hiring&sort=new&limit={limit}"
    )

    for target_url in search_targets:
        headers = {
            "User-Agent": random.choice(STEALTH_USER_AGENTS),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Polite adaptive jitter
        time.sleep(random.uniform(1.5, 3.0))

        # Exponential backoff retry loop
        max_retries = 2
        data = None

        for attempt in range(max_retries):
            try:
                resp = requests.get(target_url, headers=headers, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 429:
                    backoff = 2 ** (attempt + 1) + random.uniform(1.0, 2.5)
                    logger.warning(f"Reddit 429 rate limit hit. Backing off for {backoff:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                else:
                    logger.debug(f"Reddit JSON returned HTTP {resp.status_code} for {target_url}")
                    break
            except Exception as e:
                logger.debug(f"Request error for {target_url}: {e}")
                time.sleep(1.0)

        if not data or not isinstance(data, dict):
            continue

        children = data.get("data", {}).get("children", [])
        for child in children:
            post_data = child.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            created_utc = post_data.get("created_utc")
            permalink = post_data.get("permalink", "")

            full_text = f"{title}\n\n{selftext}".strip()
            if len(full_text) < 20 or full_text in seen_texts:
                continue

            seen_texts.add(full_text)
            posted_at = datetime.fromtimestamp(created_utc) if created_utc else datetime.utcnow()

            reviews.append({
                "company": company,
                "source": "reddit_public_json",
                "text": full_text,
                "posted_at": posted_at,
            })

            # Fetch top comment from thread if permalink available
            if permalink and len(reviews) < limit:
                try:
                    time.sleep(1.0)
                    thread_url = f"https://www.reddit.com{permalink}.json?limit=3"
                    t_resp = requests.get(thread_url, headers=headers, timeout=8)
                    if t_resp.status_code == 200:
                        t_data = t_resp.json()
                        if isinstance(t_data, list) and len(t_data) > 1:
                            comments_children = t_data[1].get("data", {}).get("children", [])
                            for c in comments_children[:3]:
                                c_body = c.get("data", {}).get("body", "")
                                if c_body and len(c_body) > 25 and c_body not in seen_texts:
                                    seen_texts.add(c_body)
                                    c_time = c.get("data", {}).get("created_utc")
                                    reviews.append({
                                        "company": company,
                                        "source": "reddit_public_json",
                                        "text": f"Comment on '{title}': {c_body}",
                                        "posted_at": datetime.fromtimestamp(c_time) if c_time else datetime.utcnow(),
                                    })
                except Exception:
                    pass

    logger.info(f"Reddit Public JSON extracted {len(reviews)} community snippets for '{company}'.")
    return reviews


def public_community_scraper(
    company: str,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    """Method B: Alternative public search & discussion forum scraper for candidate interview feedback.

    Extracts public forum search snippets where applicants discuss company hiring practices.
    """
    reviews: List[Dict[str, Any]] = []
    logger.info(f"Extracting public community forum discussions for '{company}'...")

    search_query = f'site:reddit.com "{company}" ("ghosted" OR "interview" OR "never heard back" OR "fake job")'
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"

    headers = {
        "User-Agent": random.choice(STEALTH_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.select(".result__snippet, .result__body")

            for res in results[:limit]:
                snippet_text = res.get_text(strip=True)
                if len(snippet_text) > 30 and company.lower() in snippet_text.lower():
                    reviews.append({
                        "company": company,
                        "source": "public_forum_search",
                        "text": snippet_text,
                        "posted_at": datetime.utcnow(),
                    })
    except Exception as e:
        logger.debug(f"Public community forum search error: {e}")

    logger.info(f"Public forum search scraper collected {len(reviews)} snippets for '{company}'.")
    return reviews


def reddit_scraper(
    companies: List[str],
    subreddits: List[str] = ["recruitinghell", "jobs", "antiwork"],
    limit: int = 50,
    praw_client: Any = None,
) -> List[Dict[str, Any]]:
    """Scrape candidate hiring feedback across Reddit.
    Uses authenticated PRAW if client is provided or credentials are set.
    Automatically falls back to Zero-API Public JSON and Forum Scraping otherwise.
    """
    reddit = praw_client if praw_client is not None else get_praw_client()
    reviews: List[Dict[str, Any]] = []

    if reddit is not None:
        # Authenticated or Mocked PRAW path
        for company in companies:
            logger.info(f"Searching Reddit (PRAW) for candidate feedback regarding '{company}'...")
            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    search_results = subreddit.search(company, limit=limit, sort="new")

                    for submission in search_results:
                        title = getattr(submission, "title", "")
                        selftext = getattr(submission, "selftext", "")
                        full_text = f"{title}\n\n{selftext}".strip()

                        created_utc = getattr(submission, "created_utc", None)
                        posted_at = datetime.fromtimestamp(created_utc) if created_utc else datetime.utcnow()

                        if full_text and len(full_text) > 15:
                            reviews.append({
                                "company": company,
                                "source": "reddit",
                                "text": full_text,
                                "posted_at": posted_at,
                            })

                        # Top-level comments
                        if hasattr(submission, "comments") and submission.comments:
                            comments_obj = submission.comments
                            if hasattr(comments_obj, "replace_more"):
                                try:
                                    comments_obj.replace_more(limit=0)
                                except Exception:
                                    pass
                            try:
                                top_comments = list(comments_obj)[:5]
                            except Exception:
                                top_comments = []

                            for comment in top_comments:
                                comment_body = getattr(comment, "body", "")
                                c_created = getattr(comment, "created_utc", None)
                                c_posted_at = datetime.fromtimestamp(c_created) if c_created else datetime.utcnow()

                                if comment_body and len(comment_body) > 15:
                                    reviews.append({
                                        "company": company,
                                        "source": "reddit",
                                        "text": f"Comment on '{title}': {comment_body}",
                                        "posted_at": c_posted_at,
                                    })
                except Exception as e:
                    logger.error(f"Error scraping Reddit r/{sub_name} for '{company}': {e}")
        return reviews

    # Zero-API Path: Method A (Public JSON) -> Method B (Public Web Search Fallback)
    for company in companies:
        logger.info(f"Running Zero-API candidate sentiment extraction for '{company}'...")
        comp_reviews = reddit_public_json_scraper(company=company, limit=limit, subreddits=subreddits)

        # Fallback to Method B if JSON returned 0 results
        if not comp_reviews:
            comp_reviews = public_community_scraper(company=company, limit=limit)

        reviews.extend(comp_reviews)

    logger.info(f"Reddit scraper collected {len(reviews)} total review entries across {len(companies)} companies.")
    return reviews


def glassdoor_scraper(company: str, allow_public_snippets: bool = False) -> List[Dict[str, Any]]:
    """Glassdoor scraper handler.
    Default (allow_public_snippets=False): Returns empty list stub to comply with strict site ToS.
    If allow_public_snippets=True: Extracts public syndicated interview experience summaries.
    """
    if not allow_public_snippets:
        logger.info(
            f"Glassdoor scraper called for '{company}'. Returning empty list stub (ToS compliance)."
        )
        return []

    # Safe extraction of public employer interview feedback summaries
    reviews = []
    try:
        search_q = f'"{company}" "interview review" "ghosted" OR "no offer" site:glassdoor.com'
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_q)}"
        headers = {"User-Agent": random.choice(STEALTH_USER_AGENTS)}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for res in soup.select(".result__snippet")[:5]:
                txt = res.get_text(strip=True)
                if len(txt) > 25:
                    reviews.append({
                        "company": company,
                        "source": "glassdoor_public_summary",
                        "text": txt,
                        "posted_at": datetime.utcnow(),
                    })
    except Exception as e:
        logger.debug(f"Glassdoor snippet search error: {e}")

    return reviews


def save_reviews(
    reviews: List[Dict[str, Any]],
    db_path: str = "data/ghostjobs.db",
    auto_update_sentiment: bool = True,
) -> Dict[str, int]:
    """Save reviews into CompanyReview table, deduplicating existing entries.
    Automatically updates CompanySentiment table for affected companies.
    """
    if not reviews:
        return {"total": 0, "saved": 0, "skipped": 0}

    init_db(db_path)
    session: Session = get_session(db_path)

    saved_count = 0
    skipped_count = 0
    affected_companies = set()

    try:
        for r_data in reviews:
            comp = r_data.get("company", "Unknown")
            text = r_data.get("text", "")
            source = r_data.get("source", "reddit")
            posted_at = r_data.get("posted_at", datetime.utcnow())

            if not text or len(text.strip()) == 0:
                continue

            # Deduplicate by company + text
            existing = (
                session.query(CompanyReview)
                .filter(CompanyReview.company == comp, CompanyReview.text == text)
                .first()
            )

            if existing:
                skipped_count += 1
            else:
                review_obj = CompanyReview(
                    company=comp,
                    source=source,
                    text=text,
                    posted_at=posted_at,
                    scraped_at=datetime.utcnow(),
                )
                session.add(review_obj)
                saved_count += 1
                affected_companies.add(comp)

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving reviews to DB: {e}")
        raise e
    finally:
        session.close()

    # Automatically recompute and update CompanySentiment in DB
    if auto_update_sentiment and affected_companies:
        try:
            from sentiment.analyze import company_sentiment_score

            s_session = get_session(db_path)
            for c_name in affected_companies:
                res = company_sentiment_score(c_name, db_path=db_path)
                c_score = res.get("score", 0.0)
                c_count = res.get("review_count", 0)

                existing_sent = s_session.query(CompanySentiment).filter_by(company=c_name).first()
                if existing_sent:
                    existing_sent.score = c_score
                    existing_sent.review_count = c_count
                    existing_sent.computed_at = datetime.utcnow()
                else:
                    new_sent = CompanySentiment(
                        company=c_name,
                        score=c_score,
                        review_count=c_count,
                        computed_at=datetime.utcnow(),
                    )
                    s_session.add(new_sent)
            s_session.commit()
            s_session.close()
            logger.info(f"Updated CompanySentiment scores for {len(affected_companies)} companies.")
        except Exception as e:
            logger.warning(f"Could not auto-update company sentiments: {e}")

    summary = {
        "total": len(reviews),
        "saved": saved_count,
        "skipped": skipped_count,
    }
    logger.info(f"Review save operation completed: {summary}")
    return summary
