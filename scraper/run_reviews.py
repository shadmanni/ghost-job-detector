import argparse
import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from typing import Dict, Any, List

from scraper.reviews import reddit_scraper, glassdoor_scraper, save_reviews
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
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "companies" in data:
        return data["companies"]
    elif isinstance(data, list):
        return data
    else:
        logger.error("Invalid YAML format: expected top-level 'companies' list.")
        return []


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
        "--limit",
        type=int,
        default=25,
        help="Max posts per subreddit to search per company",
    )

    args = parser.parse_args()

    init_db(args.db_path)

    companies_info = load_config(args.config)
    company_names = [c.get("name") for c in companies_info if c.get("name")]

    logger.info(f"Loaded {len(company_names)} target companies: {company_names}")

    all_reviews = []

    # 1. Reddit Scraper
    reddit_results = reddit_scraper(companies=company_names, limit=args.limit)
    all_reviews.extend(reddit_results)

    # 2. Glassdoor Stub
    for comp in company_names:
        gd_results = glassdoor_scraper(company=comp)
        all_reviews.extend(gd_results)

    # 3. Deduplicate and Save to DB
    if all_reviews:
        stats = save_reviews(all_reviews, db_path=args.db_path)
        logger.info("============================================")
        logger.info("FINAL REVIEW SCRAPE PIPELINE SUMMARY:")
        logger.info(f"  Total Review Entries Collected: {stats['total']}")
        logger.info(f"  New Reviews Saved to DB:        {stats['saved']}")
        logger.info(f"  Duplicate Reviews Skipped:      {stats['skipped']}")
        logger.info("============================================")
    else:
        logger.warning(
            "No live review entries collected. (Note: PRAW requires valid Reddit API credentials in .env to fetch live Reddit posts)."
        )


if __name__ == "__main__":
    main()
