import os
import pytest
from scraper.jobboards import (
    career_page_scraper,
    indeed_scraper,
    respect_robots_txt,
    save_postings,
)
from scraper.models import init_db, get_session, JobPosting

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def career_page_html():
    file_path = os.path.join(FIXTURES_DIR, "sample_career_page.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def indeed_html():
    file_path = os.path.join(FIXTURES_DIR, "sample_indeed.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def test_career_page_scraper_offline(career_page_html):
    """Test parsing logic for company career pages using local HTML fixture."""
    url = "https://example.com/careers"
    postings = career_page_scraper(url=url, company="Acme Corp", html_override=career_page_html)

    assert len(postings) == 3

    titles = [p["title"] for p in postings]
    assert "Senior Backend Engineer" in titles
    assert "Staff AI Researcher" in titles
    assert "Product Designer" in titles

    for p in postings:
        assert p["company"] == "Acme Corp"
        assert p["source"] == "Career Page"
        assert p["url"].startswith("https://example.com/careers/job/")

    # Verify salary extraction
    backend_job = next(p for p in postings if p["title"] == "Senior Backend Engineer")
    assert "$160,000" in backend_job["salary_listed"]


def test_indeed_scraper_offline(indeed_html):
    """Test parsing logic for Indeed search results using local HTML fixture."""
    postings = indeed_scraper(query="Developer", location="San Francisco, CA", html_override=indeed_html)

    assert len(postings) == 2

    dev_job = next(p for p in postings if p["title"] == "Full Stack Developer")
    assert dev_job["company"] == "TechCorp"
    assert dev_job["source"] == "Indeed"
    assert "jk123456789" in dev_job["url"]
    assert "$120,000" in dev_job["salary_listed"]

    data_job = next(p for p in postings if p["title"] == "Data Engineer")
    assert data_job["company"] == "DataSystems Inc"


def test_respect_robots_txt_logic(mocker=None):
    """Verify that robots.txt checking function returns boolean gracefully."""
    from unittest.mock import patch
    with patch("urllib.robotparser.RobotFileParser.read"):
        with patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=True):
            result = respect_robots_txt("https://www.example.com/page")
            assert isinstance(result, bool)
            assert result is True


def test_save_postings_repost_deduplication(tmp_path):
    """Test sentence-transformers repost detection thresholding (0.92 similarity)."""
    db_file = tmp_path / "test_dedup.db"
    db_path = str(db_file)
    init_db(db_path)

    company = "TechInnovators"

    # Posting 1: Initial Original Posting
    posting1 = [{
        "company": company,
        "title": "Senior Python Backend Developer",
        "description": "We are seeking a Senior Python Backend Developer to design, implement, and maintain high throughput web microservices using FastAPI, SQLAlchemy, and PostgreSQL in Docker.",
        "url": "https://example.com/jobs/python-1",
        "source": "Career Page",
        "salary_listed": "$150,000",
    }]

    res1 = save_postings(posting1, db_path=db_path, similarity_threshold=0.92)
    assert res1["new"] == 1
    assert res1["reposts"] == 0

    session = get_session(db_path)
    original_job = session.query(JobPosting).filter_by(url="https://example.com/jobs/python-1").first()
    assert original_job is not None
    assert original_job.repost_of_id is None
    original_id = original_job.id
    session.close()

    # Posting 2: Repost with >0.92 Near-Duplicate Description
    posting2 = [{
        "company": company,
        "title": "Senior Python Developer - Microservices",
        "description": "We are seeking a Senior Python Backend Developer to design, implement, and maintain high throughput web microservices using FastAPI, SQLAlchemy, and PostgreSQL in Docker container environment.",
        "url": "https://example.com/jobs/python-2",
        "source": "Career Page",
        "salary_listed": "$150,000",
    }]

    res2 = save_postings(posting2, db_path=db_path, similarity_threshold=0.92)
    assert res2["reposts"] == 1

    session = get_session(db_path)
    repost_job = session.query(JobPosting).filter_by(url="https://example.com/jobs/python-2").first()
    assert repost_job is not None
    assert repost_job.repost_of_id == original_id
    session.close()

    # Posting 3: Distinct Role (<0.92 Similarity)
    posting3 = [{
        "company": company,
        "title": "Lead Graphic UI Designer",
        "description": "Lead Graphic UI Designer required to design visual branding, Figma prototypes, marketing landing pages, color schemes, and icon sets for mobile applications.",
        "url": "https://example.com/jobs/designer-1",
        "source": "Career Page",
        "salary_listed": "$120,000",
    }]

    res3 = save_postings(posting3, db_path=db_path, similarity_threshold=0.92)
    assert res3["new"] == 1
    assert res3["reposts"] == 0

    session = get_session(db_path)
    distinct_job = session.query(JobPosting).filter_by(url="https://example.com/jobs/designer-1").first()
    assert distinct_job is not None
    assert distinct_job.repost_of_id is None
    session.close()
