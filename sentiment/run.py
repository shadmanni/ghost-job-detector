import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.orm import Session
from scraper.models import (
    init_db,
    get_session,
    CompanyReview,
    JobPosting,
    CompanySentiment,
)
from sentiment.analyze import company_sentiment_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentiment.run")


def process_company_sentiments(db_path: str = "data/ghostjobs.db") -> int:
    """Batch process all companies in SQLite database, computing and persisting CompanySentiment scores."""
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        # Collect distinct companies across reviews and job postings
        review_companies = [r[0] for r in session.query(CompanyReview.company).distinct().all()]
        job_companies = [j[0] for j in session.query(JobPosting.company).distinct().all()]
        distinct_companies = sorted(list(set(review_companies + job_companies)))

        logger.info(f"Processing sentiment analysis for {len(distinct_companies)} target companies in '{db_path}'...")

        if not distinct_companies:
            return 0

        saved_count = 0
        for comp in distinct_companies:
            res = company_sentiment_score(comp, db_path=db_path)
            score = res["score"]
            review_count = res["review_count"]

            # Upsert into CompanySentiment table
            existing = session.query(CompanySentiment).filter_by(company=comp).first()
            if existing:
                existing.score = score
                existing.review_count = review_count
                existing.computed_at = datetime.utcnow()
            else:
                new_sentiment = CompanySentiment(
                    company=comp,
                    score=score,
                    review_count=review_count,
                    computed_at=datetime.utcnow(),
                )
                session.add(new_sentiment)
            saved_count += 1
            logger.info(f"CompanySentiment stored for '{comp}': score={score:.4f}, reviews={review_count}")

        session.commit()
        logger.info(f"Successfully committed sentiment scores for {saved_count} companies to database.")
        return saved_count
    except Exception as e:
        session.rollback()
        logger.error(f"Error during batch sentiment analysis: {e}")
        raise e
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Sentiment Analysis Batch CLI")
    parser.add_argument(
        "--db-path",
        default="data/ghostjobs.db",
        help="Path to SQLite database file",
    )

    args = parser.parse_args()
    process_company_sentiments(db_path=args.db_path)


if __name__ == "__main__":
    main()
