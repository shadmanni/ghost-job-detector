import os
import pytest
from datetime import datetime, timedelta
from scraper.models import init_db, get_session, PageAnalytics, JobPosting, GhostScore
from analytics.ga4_client import (
    fetch_ga4_30day_page_metrics,
    sync_ga4_page_analytics_to_db,
    is_ga4_configured,
)
from seo.build import build_site


def test_page_analytics_model(tmp_path):
    """Test creating and querying PageAnalytics database records."""
    db_path = str(tmp_path / "test_analytics.db")
    init_db(db_path)
    session = get_session(db_path)

    try:
        analytics = PageAnalytics(
            company="Anthropic",
            pageviews=1200,
            avg_time_on_page=145.5,
            bounce_rate=0.35,
        )
        session.add(analytics)
        session.commit()

        queried = session.query(PageAnalytics).filter_by(company="Anthropic").first()
        assert queried is not None
        assert queried.pageviews == 1200
        assert queried.avg_time_on_page == 145.5
        assert queried.bounce_rate == 0.35
    finally:
        session.close()


def test_fetch_ga4_30day_page_metrics_unconfigured(monkeypatch):
    """Test fetch_ga4_30day_page_metrics returns empty list when unconfigured."""
    monkeypatch.delenv("GA4_PROPERTY_ID", raising=False)
    monkeypatch.delenv("GA4_MEASUREMENT_ID", raising=False)
    assert fetch_ga4_30day_page_metrics() == []


def test_fetch_ga4_30day_page_metrics_configured(monkeypatch):
    """Test fetch_ga4_30day_page_metrics returns structured metrics when configured."""
    monkeypatch.setenv("GA4_MEASUREMENT_ID", "G-999999999")
    metrics = fetch_ga4_30day_page_metrics()
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    item = metrics[0]
    assert "company" in item
    assert "pageviews" in item
    assert "avg_time_on_page" in item
    assert "bounce_rate" in item


def test_sync_ga4_page_analytics_to_db(tmp_path, monkeypatch):
    """Test syncing GA4 metrics to SQLite database cache."""
    db_path = str(tmp_path / "test_sync.db")
    monkeypatch.setenv("GA4_MEASUREMENT_ID", "G-888888888")

    count = sync_ga4_page_analytics_to_db(db_path=db_path, force=True)
    assert count > 0

    session = get_session(db_path)
    try:
        rows = session.query(PageAnalytics).all()
        assert len(rows) == count
    finally:
        session.close()


def test_seo_gtag_snippet_injection(tmp_path, monkeypatch):
    """Test Jinja template injects gtag.js script when GA4_MEASUREMENT_ID is set."""
    db_path = str(tmp_path / "test_seo_ga4.db")
    init_db(db_path)
    session = get_session(db_path)
    try:
        posting = JobPosting(company="TestCorp", title="Engineer", url="https://example.com")
        session.add(posting)
        session.commit()

        score = GhostScore(job_posting_id=posting.id, final_score=75.0)
        session.add(score)
        session.commit()
    finally:
        session.close()

    # 1. Without GA4_MEASUREMENT_ID
    monkeypatch.delenv("GA4_MEASUREMENT_ID", raising=False)
    out_dir_no_ga4 = str(tmp_path / "seo_no_ga4")
    build_site(db_path=db_path, out_dir=out_dir_no_ga4)

    index_path_1 = os.path.join(out_dir_no_ga4, "index.html")
    assert os.path.exists(index_path_1)
    with open(index_path_1, "r", encoding="utf-8") as f:
        html_1 = f.read()
    assert "googletagmanager.com/gtag/js" not in html_1

    # 2. With GA4_MEASUREMENT_ID
    monkeypatch.setenv("GA4_MEASUREMENT_ID", "G-REAL123456")
    out_dir_ga4 = str(tmp_path / "seo_ga4")
    build_site(db_path=db_path, out_dir=out_dir_ga4)

    index_path_2 = os.path.join(out_dir_ga4, "index.html")
    assert os.path.exists(index_path_2)
    with open(index_path_2, "r", encoding="utf-8") as f:
        html_2 = f.read()
    assert "googletagmanager.com/gtag/js?id=G-REAL123456" in html_2
    assert "gtag('config', 'G-REAL123456');" in html_2
