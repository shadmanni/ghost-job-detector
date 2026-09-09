from datetime import datetime
from unittest.mock import MagicMock
import pytest

from scraper.reviews import reddit_scraper, glassdoor_scraper, save_reviews
from scraper.models import init_db, get_session, CompanyReview


def test_reddit_scraper_mocked():
    """Test Reddit scraper parsing using mocked PRAW client responses (no network calls)."""
    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()

    # Mock Submission object
    mock_submission = MagicMock()
    mock_submission.title = "Anthropic ghost job experience"
    mock_submission.selftext = "I applied 3 months ago and got no response despite reposts."
    mock_submission.created_utc = 1700000000.0

    # Mock Comment object
    mock_comment = MagicMock()
    mock_comment.body = "Same thing happened to me! Total ghost posting."
    mock_comment.created_utc = 1700000500.0

    mock_submission.comments = [mock_comment]
    mock_subreddit.search.return_value = [mock_submission]
    mock_reddit.subreddit.return_value = mock_subreddit

    reviews = reddit_scraper(
        companies=["Anthropic"],
        subreddits=["recruitinghell"],
        limit=10,
        praw_client=mock_reddit,
    )

    assert len(reviews) == 2

    post_review = reviews[0]
    assert post_review["company"] == "Anthropic"
    assert post_review["source"] == "reddit"
    assert "Anthropic ghost job experience" in post_review["text"]
    assert isinstance(post_review["posted_at"], datetime)

    comment_review = reviews[1]
    assert comment_review["company"] == "Anthropic"
    assert comment_review["source"] == "reddit"
    assert "Total ghost posting" in comment_review["text"]


def test_glassdoor_scraper_stub():
    """Verify that Glassdoor scraper stub logs decision and returns safe empty list."""
    results = glassdoor_scraper("OpenAI")
    assert isinstance(results, list)
    assert len(results) == 0


def test_save_reviews_deduplication(tmp_path):
    """Test saving reviews and verifying exact content deduplication in SQLite database."""
    db_file = tmp_path / "test_reviews.db"
    db_path = str(db_file)
    init_db(db_path)

    company = "OpenAI"

    reviews_data = [
        {
            "company": company,
            "source": "reddit",
            "text": "Interview process took 6 rounds then radio silence.",
            "posted_at": datetime.utcnow(),
        },
        {
            "company": company,
            "source": "reddit",
            "text": "Great company culture feedback on Reddit.",
            "posted_at": datetime.utcnow(),
        },
    ]

    # First save operation
    stats1 = save_reviews(reviews_data, db_path=db_path)
    assert stats1["saved"] == 2
    assert stats1["skipped"] == 0

    session = get_session(db_path)
    db_reviews = session.query(CompanyReview).filter_by(company=company).all()
    assert len(db_reviews) == 2
    session.close()

    # Second save operation (duplicate contents)
    stats2 = save_reviews(reviews_data, db_path=db_path)
    assert stats2["saved"] == 0
    assert stats2["skipped"] == 2

    session = get_session(db_path)
    db_reviews_after = session.query(CompanyReview).filter_by(company=company).all()
    assert len(db_reviews_after) == 2
    session.close()
