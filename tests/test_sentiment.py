from datetime import datetime, timedelta
import pytest

from sentiment.analyze import (
    baseline_sentiment,
    frustration_score,
    company_sentiment_score,
)
from scraper.models import init_db, get_session, CompanyReview


def test_frustration_score_high_vs_low():
    """Verify that frustrated candidate review text scores clearly higher than neutral/positive review text."""
    frustrated_text = (
        "I was completely ghosted after the final round interview. It turned out to be a fake job posting "
        "and a complete waste of time. I never heard back from the recruiter despite radio silence for 2 months."
    )

    neutral_text = (
        "The software engineering interview process consisted of a coding assessment and a system design discussion. "
        "The interviewers were professional and answered all my questions about team culture."
    )

    frustrated_score = frustration_score(frustrated_text)
    neutral_score = frustration_score(neutral_text)

    assert frustrated_score > 0.60
    assert neutral_score < 0.30
    assert frustrated_score > neutral_score + 0.35


def test_baseline_sentiment():
    """Test NLTK VADER baseline sentiment calculation."""
    neg_text = "Terrible experience, unresponsive recruiters, complete disaster."
    pos_text = "Outstanding culture, wonderful leadership, excellent compensation package."

    neg_score = baseline_sentiment(neg_text)
    pos_score = baseline_sentiment(pos_text)

    assert neg_score > pos_score


def test_company_sentiment_score_recency_weighting(tmp_path):
    """Verify recency-weighted company sentiment score calculation on SQLite test DB."""
    db_path = str(tmp_path / "test_sentiment.db")
    init_db(db_path)
    session = get_session(db_path)

    company = "TechCorp"
    now = datetime.utcnow()

    # Recent highly frustrated review (1 day ago)
    r1 = CompanyReview(
        company=company,
        source="reddit",
        text="Ghosted after final round. Never heard back.",
        posted_at=now - timedelta(days=1),
    )

    # Older neutral review (60 days ago)
    r2 = CompanyReview(
        company=company,
        source="reddit",
        text="Standard interview process.",
        posted_at=now - timedelta(days=60),
    )

    session.add(r1)
    session.add(r2)
    session.commit()
    session.close()

    res = company_sentiment_score(company, db_path=db_path)

    assert res["company"] == company
    assert res["review_count"] == 2
    # Because recent review is heavily frustrated, recency-weighted mean should lean towards high score
    assert res["score"] > 0.40
