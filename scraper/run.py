import argparse
import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from typing import Dict, Any, List

from scraper.jobboards import career_page_scraper, indeed_scraper, save_postings
from scraper.models import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scraper.run")


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
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Scraping CLI Pipeline")
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
        "--source",
        choices=["career", "indeed", "all"],
        default="all",
        help="Scraping source engine to execute ('career', 'indeed', or 'all')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of companies to scrape from config",
    )

    args = parser.parse_args()

    # Ensure DB tables exist
    init_db(args.db_path)

    companies = load_config(args.config)
    if args.limit:
        companies = companies[: args.limit]

    logger.info(f"Loaded {len(companies)} target companies from '{args.config}'")

    total_stats = {"total": 0, "new": 0, "reposts": 0, "skipped": 0}

    for comp_info in companies:
        name = comp_info.get("name", "Unknown")
        career_url = comp_info.get("career_url")
        query = comp_info.get("query", "Software Engineer")
        location = comp_info.get("location", "San Francisco, CA")

        logger.info(f"=== Processing Company: {name} ===")

        collected_postings = []

        # 1. Career Page Scraper
        if args.source in ["career", "all"] and career_url:
            try:
                c_postings = career_page_scraper(url=career_url, company=name)
                collected_postings.extend(c_postings)
            except Exception as e:
                logger.error(f"Career page scrape failed for {name}: {e}")

        # 2. Indeed Scraper (Fallback/Secondary)
        if args.source in ["indeed", "all"]:
            try:
                i_postings = indeed_scraper(query=query, location=location, company=name)
                collected_postings.extend(i_postings)
            except Exception as e:
                logger.error(f"Indeed scrape failed for {name}: {e}")

        # 3. Save & Deduplicate Postings
        if collected_postings:
            stats = save_postings(collected_postings, db_path=args.db_path)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
            logger.info(
                f"Summary for {name}: Total Extracted: {stats['total']} | "
                f"New Postings: {stats['new']} | Reposts Detected: {stats['reposts']} | "
                f"Skipped Duplicates: {stats['skipped']}"
            )
        else:
            logger.warning(f"No postings extracted for {name}")

    logger.info("============================================")
    logger.info("FINAL SCRAPE PIPELINE SUMMARY:")
    logger.info(f"  Total Postings Extracted: {total_stats['total']}")
    logger.info(f"  New Postings Created:    {total_stats['new']}")
    logger.info(f"  Reposts Flagged (FK set): {total_stats['reposts']}")
    logger.info(f"  Exact Duplicates Skipped: {total_stats['skipped']}")
    logger.info("============================================")


if __name__ == "__main__":
    main()
