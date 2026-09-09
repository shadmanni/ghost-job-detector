import os
import pytest
from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanyReview, PageAnalytics
from scripts.export_for_paper import (
    export_classifier_metrics,
    export_sentiment_correlation_plot,
    export_ga4_correlation_plot,
    export_summary_stats_markdown,
)


def test_export_classifier_metrics(tmp_path):
    """Test exporting classifier evaluation metrics and confusion matrix plot."""
    out_dir = str(tmp_path / "paper_reports")
    metrics = export_classifier_metrics(out_dir=out_dir)

    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics

    csv_file = os.path.join(out_dir, "classifier_metrics.csv")
    png_file = os.path.join(out_dir, "confusion_matrix.png")
    assert os.path.exists(csv_file)
    assert os.path.exists(png_file)


def test_export_sentiment_correlation_plot(tmp_path):
    """Test exporting sentiment correlation plot."""
    db_path = str(tmp_path / "test_export.db")
    out_dir = str(tmp_path / "paper_reports")
    init_db(db_path)
    session = get_session(db_path)

    try:
        posting = JobPosting(company="Anthropic", title="AI Researcher", url="https://example.com")
        session.add(posting)
        session.commit()

        ghost_score = GhostScore(
            job_posting_id=posting.id,
            sentiment_score=0.8,
            final_score=85.0,
        )
        session.add(ghost_score)
        session.commit()
    finally:
        session.close()

    res = export_sentiment_correlation_plot(db_path=db_path, out_dir=out_dir)
    assert "pearson_r" in res
    assert "p_value" in res

    png_file = os.path.join(out_dir, "sentiment_vs_ghost_score.png")
    assert os.path.exists(png_file)


def test_export_ga4_correlation_plot(tmp_path):
    """Test exporting GA4 correlation plot."""
    db_path = str(tmp_path / "test_export.db")
    out_dir = str(tmp_path / "paper_reports")
    init_db(db_path)

    res = export_ga4_correlation_plot(db_path=db_path, out_dir=out_dir)
    assert "pearson_r" in res
    assert "p_value" in res

    png_file = os.path.join(out_dir, "ga4_traffic_vs_ghost_score.png")
    assert os.path.exists(png_file)


def test_export_summary_stats_markdown(tmp_path):
    """Test exporting IEEE summary stats markdown file."""
    db_path = str(tmp_path / "test_export.db")
    out_dir = str(tmp_path / "paper_reports")
    init_db(db_path)

    md_text = export_summary_stats_markdown(db_path=db_path, out_dir=out_dir)
    assert "Empirical Results & Dataset Summary Statistics" in md_text
    assert "Ghost Job Score Statistical Distribution" in md_text
    assert "Classifier Performance Metrics" in md_text

    md_file = os.path.join(out_dir, "summary_stats.md")
    assert os.path.exists(md_file)
