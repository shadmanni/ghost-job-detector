import argparse
import logging
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.orm import Session
from scraper.models import init_db, get_session, JobPosting
from preprocessing.clean import clean_job_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocessing.run")


def process_uncleaned_postings(db_path: str = "data/ghostjobs.db", force: bool = False) -> int:
    """Batch process JobPosting rows in SQLite database, applying clean_job_text and storing cleaned_text."""
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        if force:
            postings = session.query(JobPosting).all()
        else:
            postings = session.query(JobPosting).filter(JobPosting.cleaned_text == None).all()

        count = len(postings)
        logger.info(f"Found {count} JobPosting rows to clean in '{db_path}' (force={force})")

        if count == 0:
            return 0

        processed = 0
        for job in postings:
            raw_text = job.description or ""
            cleaned = clean_job_text(raw_text)
            job.cleaned_text = cleaned
            processed += 1

        session.commit()
        logger.info(f"Successfully cleaned and updated {processed} JobPosting rows in database.")
        return processed
    except Exception as e:
        session.rollback()
        logger.error(f"Error during batch text cleaning: {e}")
        raise e
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Text Preprocessing Batch CLI")
    parser.add_argument(
        "--db-path",
        default="data/ghostjobs.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-cleaning of all postings including previously cleaned rows",
    )

    args = parser.parse_args()
    process_uncleaned_postings(db_path=args.db_path, force=args.force)


if __name__ == "__main__":
    main()
