import argparse
import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from typing import Dict, Any, List

from scraper.reviews import (
    reddit_scraper,
    glassdoor_scraper,
    save_reviews,
    public_community_scraper,
)
from scraper.models import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scraper.run_reviews")


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """Load target companies configuration from a YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "companies" in data:
        return data["companies"]
    elif isinstance(data, list):
        return data
    else:
        logger.error("Invalid YAML format: expected top-level 'companies' list.")
        return []


def run_reviews_for_company(
    company: str,
    limit: int = 25,
    db_path: str = "data/ghostjobs.db",
) -> Dict[str, Any]:
    """Execute Zero-API community sentiment extraction for a single company."""
    logger.info(f"Extracting community sentiment & hiring feedback for '{company}'...")
    all_reviews = []

    # 1. Zero-API Reddit scraper (Public JSON suffix with backoff + Web search fallback)
    reddit_results = reddit_scraper(companies=[company], limit=limit)
    all_reviews.extend(reddit_results)

    # 2. Public Community Discussion Search
    if len(reddit_results) < 5:
        comm_results = public_community_scraper(company=company, limit=10)
        all_reviews.extend(comm_results)

    # 3. Deduplicate and Save to DB + auto-update CompanySentiment
    stats = {"total": 0, "saved": 0, "skipped": 0}
    if all_reviews:
        stats = save_reviews(all_reviews, db_path=db_path, auto_update_sentiment=True)

    return {"company": company, "reviews": len(all_reviews), "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Sentiment/Review Scraper CLI")
    parser.add_argument(
        "--config",
        default="config/target_companies.yaml",
        help="Path to YAML config file with target companies",
    )
    parser.add_argument(
        "--db-path",
        default="data/ghostjobs.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Target single company on-demand",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max posts per subreddit to search per company",
    )

    args = parser.parse_args()

    init_db(args.db_path)

    if args.company:
        company_names = [args.company]
    else:
        companies_info = load_config(args.config)
        company_names = [c.get("name") for c in companies_info if c.get("name")]

    logger.info(f"Targeting {len(company_names)} companies for community sentiment: {company_names}")

    total_stats = {"total": 0, "saved": 0, "skipped": 0}

    for comp in company_names:
        res = run_reviews_for_company(company=comp, limit=args.limit, db_path=args.db_path)
        for k in total_stats:
            total_stats[k] += res["stats"].get(k, 0)

    logger.info("============================================")
    logger.info("FINAL REVIEW SCRAPE PIPELINE SUMMARY:")
    logger.info(f"  Total Review Entries Collected: {total_stats['total']}")
    logger.info(f"  New Reviews Saved to DB:        {total_stats['saved']}")
    logger.info(f"  Duplicate Reviews Skipped:      {total_stats['skipped']}")
    logger.info("============================================")


if __name__ == "__main__":
    main()
