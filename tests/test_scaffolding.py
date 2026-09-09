import os
import pytest
from sqlalchemy import inspect
from scraper.models import (
    JobPosting,
    CompanyReview,
    GhostScore,
    init_db,
    get_session,
    Base,
)


def test_models_importable():
    """Verify that all core ORM models are imported correctly."""
    assert JobPosting is not None
    assert CompanyReview is not None
    assert GhostScore is not None


def test_database_initialization(tmp_path):
    """Verify that SQLite database tables are created with expected schema."""
    db_file = tmp_path / "test_ghostjobs.db"
    engine = init_db(str(db_file))
    assert db_file.exists()

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    assert "job_postings" in table_names
    assert "company_reviews" in table_names
    assert "ghost_scores" in table_names


def test_model_creation_and_query(tmp_path):
    """Verify CRUD operations on ORM models in SQLite test database."""
    db_file = tmp_path / "test_ghostjobs.db"
    init_db(str(db_file))
    session = get_session(str(db_file))

    try:
        # Create a job posting
        job = JobPosting(
            company="Acme Corp",
            title="Senior Software Engineer",
            description="We are looking for a visionary developer...",
            url="https://example.com/jobs/123",
            source="LinkedIn",
            salary_listed="$150k - $180k",
        )
        session.add(job)
        session.commit()

        assert job.id is not None

        # Create a ghost score for the posting
        score = GhostScore(
            job_posting_id=job.id,
            genericness_score=0.85,
            vagueness_score=0.72,
            repost_score=0.10,
            urgency_score=0.90,
            bert_score=0.80,
            sentiment_score=0.65,
            final_score=0.79,
        )
        session.add(score)
        session.commit()

        # Query back
        queried_job = session.query(JobPosting).filter_by(company="Acme Corp").first()
        assert queried_job is not None
        assert len(queried_job.scores) == 1
        assert queried_job.scores[0].final_score == pytest.approx(0.79)
    finally:
        session.close()
