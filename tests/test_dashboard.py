import os
import yaml
import pytest
import pandas as pd
from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanySentiment
from analytics.ga4_client import get_ga4_traffic_metrics
from dashboard.app import load_company_industries, load_dashboard_data


def test_ga4_client_unconfigured(monkeypatch):
    """Test GA4 client fallback when env vars are unconfigured or placeholders."""
    monkeypatch.setenv("GA4_MEASUREMENT_ID", "your_measurement_id_here")
    monkeypatch.setenv("GA4_API_SECRET", "your_api_secret_here")
    assert get_ga4_traffic_metrics() is None

    monkeypatch.delenv("GA4_MEASUREMENT_ID", raising=False)
    monkeypatch.delenv("GA4_API_SECRET", raising=False)
    assert get_ga4_traffic_metrics() is None


def test_ga4_client_configured(monkeypatch):
    """Test GA4 client return structure when configured with valid keys."""
    monkeypatch.setenv("GA4_MEASUREMENT_ID", "G-123456789")
    monkeypatch.setenv("GA4_API_SECRET", "secret_abc123")
    metrics = get_ga4_traffic_metrics()
    assert metrics is not None
    assert metrics.get("connected") is True
    assert "active_users_30d" in metrics
    assert "top_reports" in metrics


def test_load_company_industries(tmp_path):
    """Test reading company industry mappings from YAML config."""
    config_file = tmp_path / "test_target_companies.yaml"
    data = {
        "companies": [
            {"name": "Anthropic", "industry": "Artificial Intelligence"},
            {"name": "Stripe", "industry": "Fintech"},
        ]
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    mapping = load_company_industries(str(config_file))
    assert mapping["Anthropic"] == "Artificial Intelligence"
    assert mapping["Stripe"] == "Fintech"

    # Missing file returns empty dict
    missing_mapping = load_company_industries(str(tmp_path / "nonexistent.yaml"))
    assert missing_mapping == {}


def test_load_dashboard_data(tmp_path):
    """Test querying live data from SQLite DB for dashboard."""
    db_path = str(tmp_path / "test_dashboard.db")
    init_db(db_path)
    session = get_session(db_path)

    try:
        posting = JobPosting(
            company="TestCorp",
            title="Senior Engineer",
            source="career_page",
            url="https://example.com/job/1",
        )
        session.add(posting)
        session.commit()

        ghost_score = GhostScore(
            job_posting_id=posting.id,
            genericness_score=0.3,
            vagueness_score=0.4,
            repost_score=0.1,
            urgency_score=0.2,
            bert_score=0.5,
            sentiment_score=0.6,
            final_score=45.0,
        )
        sentiment = CompanySentiment(
            company="TestCorp",
            score=0.6,
            review_count=10,
        )
        session.add(ghost_score)
        session.add(sentiment)
        session.commit()
    finally:
        session.close()

    df_postings, df_sentiments, df_analytics = load_dashboard_data(db_path)
    assert not df_postings.empty
    assert len(df_postings) == 1
    assert df_postings.iloc[0]["company"] == "TestCorp"
    assert df_postings.iloc[0]["final_ghost_score"] == 45.0

    assert not df_sentiments.empty
    assert len(df_sentiments) == 1
    assert df_sentiments.iloc[0]["company"] == "TestCorp"
