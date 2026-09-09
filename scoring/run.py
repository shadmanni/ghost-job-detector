import argparse
import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy.orm import Session
from scraper.models import init_db, get_session, JobPosting, GhostScore
from scoring.final_score import compute_and_store_scores, correlation_check, DEFAULT_SCORE_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scoring.run")


def print_score_breakdown(posting: JobPosting, score: GhostScore, rank_label: str):
    """Format and print component score breakdown for a posting."""
    print(f"\n{rank_label} | Posting ID {posting.id}: '{posting.title}' ({posting.company})")
    print(f"  URL: {posting.url or 'N/A'}")
    print(f"  --> FINAL GHOST SCORE: {score.final_score:.2f} / 100.0")
    print("  [Component Sub-Score Breakdown]:")
    print(f"    - Genericness Score: {score.genericness_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['genericness']:.2f})")
    print(f"    - Vagueness Score:    {score.vagueness_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['vagueness']:.2f})")
    print(f"    - Repost Score:       {score.repost_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['repost']:.2f})")
    print(f"    - Urgency Score:      {score.urgency_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['urgency']:.2f})")
    print(f"    - Classifier (BERT):  {score.bert_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['bert_classifier']:.2f})")
    print(f"    - Company Sentiment:  {score.sentiment_score:.4f} (Weight: {DEFAULT_SCORE_WEIGHTS['company_sentiment']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - Final Scoring CLI")
    parser.add_argument(
        "--db-path",
        default="data/ghostjobs.db",
        help="Path to SQLite database file",
    )

    args = parser.parse_args()

    # 1. Compute & Store Final Ghost Scores
    count = compute_and_store_scores(db_path=args.db_path)

    # 2. Run Pearson / Spearman Correlation Check
    corr_results = correlation_check(db_path=args.db_path)

    # 3. Query & Display Top 5 Highest vs Top 5 Lowest Scoring Postings
    session: Session = get_session(args.db_path)
    try:
        query_results = (
            session.query(JobPosting, GhostScore)
            .join(GhostScore, JobPosting.id == GhostScore.job_posting_id)
            .order_by(GhostScore.final_score.desc())
            .all()
        )

        if not query_results:
            logger.warning("No scored postings found in database.")
            return

        top_5 = query_results[:5]
        bottom_5 = query_results[-5:]
        # Reverse bottom_5 so lowest-scoring are ordered from lowest upwards
        bottom_5_sorted = sorted(bottom_5, key=lambda x: x[1].final_score)

        print("\n" + "=" * 80)
        print("TOP 5 HIGHEST SCORING POSTINGS (LIKELY GHOST JOBS)")
        print("=" * 80)
        for idx, (posting, score) in enumerate(top_5, 1):
            print_score_breakdown(posting, score, f"[HIGH #{idx}]")

        print("\n" + "=" * 80)
        print("TOP 5 LOWEST SCORING POSTINGS (LIKELY LEGITIMATE JOBS)")
        print("=" * 80)
        for idx, (posting, score) in enumerate(bottom_5_sorted, 1):
            print_score_breakdown(posting, score, f"[LOW  #{idx}]")

        print("\n" + "=" * 80)
    finally:
        session.close()


if __name__ == "__main__":
    main()
