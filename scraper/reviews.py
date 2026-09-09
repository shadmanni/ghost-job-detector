from datetime import datetime
import logging
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from scraper.models import CompanyReview, init_db, get_session

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_praw_client():
    """Initialize PRAW Reddit client using environment credentials if available."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "GhostJobDetector/1.0")

    if not client_id or not client_secret or "your_" in client_id.lower():
        logger.warning(
            "Reddit API credentials (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET) not set or are default placeholders in .env. "
            "PRAW client cannot authenticate for live requests."
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


def reddit_scraper(
    companies: List[str],
    subreddits: List[str] = ["recruitinghell", "jobs", "antiwork"],
    limit: int = 50,
    praw_client: Any = None,
) -> List[Dict[str, Any]]:
    """Scrape Reddit posts and top-level comments for company hiring experiences using PRAW."""
    reddit = praw_client if praw_client is not None else get_praw_client()
    reviews: List[Dict[str, Any]] = []

    if reddit is None:
        logger.warning("No authenticated PRAW client available. Returning 0 live Reddit reviews.")
        return reviews

    for company in companies:
        logger.info(f"Searching Reddit for candidate feedback regarding '{company}'...")
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

                    # Top-level comments extraction
                    if hasattr(submission, "comments") and submission.comments:
                        comments_obj = submission.comments
                        if hasattr(comments_obj, "replace_more"):
                            try:
                                comments_obj.replace_more(limit=0)
                            except Exception as e:
                                logger.debug(f"Could not call replace_more on comments: {e}")
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

    logger.info(f"Reddit scraper collected {len(reviews)} review entries across {len(companies)} companies.")
    return reviews


def glassdoor_scraper(company: str) -> List[Dict[str, Any]]:
    """Glassdoor scraper stub.

    ================================================================================
    LEGAL & TECHNICAL DECISION ON GLASSDOOR SCRAPING:
    --------------------------------------------------------------------------------
    Glassdoor strictly prohibits automated web scraping under section 8 of its
    Terms of Use. Glassdoor deploys aggressive anti-bot protections (Cloudflare,
    DataDome, JavaScript challenges, IP rate limits) that block unauthenticated automated scrapers.

    To remain 100% compliant with web scraping ethical standards and site ToS,
    this function serves as an extensible stub. For production Glassdoor sentiment
    analysis, users should either:
    1. Import offline CSV datasets (e.g. Glassdoor data dumps).
    2. Interface with Glassdoor's official Partner API / Enterprise APIs.
    ================================================================================
    """
    logger.info(
        f"Glassdoor scraper called for '{company}'. Note: Direct Glassdoor scraping is ToS-restricted. "
        f"Returning empty list stub (use CSV import or API integration)."
    )
    return []


def save_reviews(
    reviews: List[Dict[str, Any]],
    db_path: str = "data/ghostjobs.db",
) -> Dict[str, int]:
    """Save reviews into CompanyReview table, deduplicating existing entries."""
    if not reviews:
        return {"total": 0, "saved": 0, "skipped": 0}

    init_db(db_path)
    session: Session = get_session(db_path)

    saved_count = 0
    skipped_count = 0

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

        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving reviews to DB: {e}")
        raise e
    finally:
        session.close()

    summary = {
        "total": len(reviews),
        "saved": saved_count,
        "skipped": skipped_count,
    }
    logger.info(f"Review save operation completed: {summary}")
    return summary
