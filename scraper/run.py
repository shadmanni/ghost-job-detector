import argparse
import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from typing import Dict, Any, List, Optional

from scraper.jobboards import (
    career_page_scraper,
    indeed_scraper,
    linkedin_scraper,
    save_postings,
)
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


def run_scrape_for_company(
    name: str,
    career_url: Optional[str] = None,
    query: str = "Software Engineer",
    location: str = "San Francisco, CA",
    source: str = "all",
    db_path: str = "data/ghostjobs.db",
    max_results: int = 25,
) -> Dict[str, Any]:
    """Execute live scraping across configured sources for a single company."""
    logger.info(f"=== Running Live Scrape for Company: {name} (Source: {source}) ===")
    collected_postings = []

    # 1. Career Page Scraper
    if source in ["career", "all"] and career_url:
        try:
            c_postings = career_page_scraper(url=career_url, company=name)
            collected_postings.extend(c_postings)
        except Exception as e:
            logger.error(f"Career page scrape failed for {name}: {e}")

    # 2. LinkedIn Public Scraper
    if source in ["linkedin", "all"]:
        try:
            l_postings = linkedin_scraper(
                query=query,
                location=location,
                company=name,
                limit=max_results,
            )
            collected_postings.extend(l_postings)
        except Exception as e:
            logger.error(f"LinkedIn public scrape failed for {name}: {e}")

    # 3. Indeed Scraper (Secondary / Fallback)
    if source in ["indeed", "all"]:
        try:
            i_postings = indeed_scraper(query=query, location=location, company=name)
            collected_postings.extend(i_postings)
        except Exception as e:
            logger.error(f"Indeed scrape failed for {name}: {e}")

    # 4. Save & Deduplicate Postings
    stats = {"total": 0, "new": 0, "reposts": 0, "skipped": 0}
    if collected_postings:
        stats = save_postings(collected_postings, db_path=db_path)
        logger.info(
            f"Result for {name}: Extracted={stats['total']}, New={stats['new']}, "
            f"Reposts={stats['reposts']}, Skipped={stats['skipped']}"
        )
    else:
        logger.warning(f"No postings collected for {name}")

    return {
        "company": name,
        "collected": len(collected_postings),
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Live Scraping CLI Pipeline")
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
        choices=["career", "indeed", "linkedin", "all"],
        default="all",
        help="Scraping source engine to execute ('career', 'indeed', 'linkedin', or 'all')",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Target a single specific company on-demand (bypasses config list)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Software Engineer",
        help="Target job title or search keywords for live search",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="San Francisco, CA",
        help="Target geographic location or 'Remote'",
    )
    parser.add_argument(
        "--career-url",
        type=str,
        default=None,
        help="Direct career page URL if targeting a single company",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of companies to scrape from config",
    )
    parser.add_argument(
        "--recalculate-scores",
        action="store_true",
        help="Automatically trigger preprocessing, sentiment, and scoring after scraping",
    )

    args = parser.parse_args()

    # Ensure DB tables exist
    init_db(args.db_path)

    total_stats = {"total": 0, "new": 0, "reposts": 0, "skipped": 0}

    # On-demand single company run
    if args.company:
        res = run_scrape_for_company(
            name=args.company,
            career_url=args.career_url,
            query=args.query,
            location=args.location,
            source=args.source,
            db_path=args.db_path,
        )
        for k in total_stats:
            total_stats[k] += res["stats"].get(k, 0)
    else:
        companies = load_config(args.config)
        if args.limit:
            companies = companies[: args.limit]

        logger.info(f"Loaded {len(companies)} target companies from '{args.config}'")

        for comp_info in companies:
            name = comp_info.get("name", "Unknown")
            c_url = comp_info.get("career_url")
            q = comp_info.get("query", args.query)
            loc = comp_info.get("location", args.location)

            res = run_scrape_for_company(
                name=name,
                career_url=c_url,
                query=q,
                location=loc,
                source=args.source,
                db_path=args.db_path,
            )
            for k in total_stats:
                total_stats[k] += res["stats"].get(k, 0)

    logger.info("============================================")
    logger.info("FINAL SCRAPE PIPELINE SUMMARY:")
    logger.info(f"  Total Postings Extracted: {total_stats['total']}")
    logger.info(f"  New Postings Created:    {total_stats['new']}")
    logger.info(f"  Reposts Flagged (FK set): {total_stats['reposts']}")
    logger.info(f"  Exact Duplicates Skipped: {total_stats['skipped']}")
    logger.info("============================================")

    if args.recalculate_scores:
        logger.info("Recalculating scores and updating database records...")
        from preprocessing.clean import run_preprocessing_pipeline
        from scoring.final_score import compute_and_store_scores

        run_preprocessing_pipeline(db_path=args.db_path)
        compute_and_store_scores(db_path=args.db_path)
        logger.info("Scoring recalculation complete.")


if __name__ == "__main__":
    main()
