import os
import pytest
from nlp.features import (
    genericness_score,
    vagueness_score,
    repost_score,
    urgency_score,
)
from nlp.classifier import predict, build_training_set


def test_genericness_score_high():
    """Verify that buzzword-heavy text receives a genericness score > 0.5."""
    text = (
        "We are looking for a passionate individual and self-starter to join our fast-paced environment and dynamic team. "
        "You will wear many hats in a fast-growing company with a great culture. "
        "We offer a competitive salary, market-competitive benefits, and unmatched growth opportunity."
    )
    score = genericness_score(text)
    assert score > 0.5


def test_genericness_score_low():
    """Verify that specific technical text receives a low genericness score."""
    text = (
        "The candidate will implement raft consensus in Rust, optimize SQLite B-tree page cache allocations, "
        "and maintain zero-copy network buffers over Linux epoll sockets."
    )
    score = genericness_score(text)
    assert score < 0.2


def test_vagueness_score_low_for_specific_tech():
    """Verify that specific technical nouns produce a low vagueness score (< 0.4)."""
    text = "Deploying Docker containers to AWS EKS using Terraform, Helm, Prometheus, and Grafana monitoring."
    score = vagueness_score(text)
    assert score < 0.4


def test_repost_score():
    """Verify repost score calculation for original vs reposted postings."""
    original_posting = {"repost_of_id": None, "repost_count": 0}
    reposted_posting = {"repost_of_id": 101, "repost_count": 2}

    assert repost_score(original_posting) == 0.0
    assert repost_score(reposted_posting) >= 0.5


def test_urgency_score():
    """Verify urgency score for text with evergreen and urgency phrases."""
    text = "We are urgently hiring for an immediate start. Apply now for this open pipeline talent pool."
    score = urgency_score(text)
    assert score > 0.5


def test_predict_fallback():
    """Verify predict fallback weighted average calculation."""
    text = "We are seeking a Python developer for a dynamic team. Apply now."
    prob = predict(text)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_build_training_set(tmp_path):
    """Verify build_training_set creates valid CSV output."""
    db_path = str(tmp_path / "test_nlp_db.db")
    csv_path = str(tmp_path / "test_output.csv")

    from scraper.models import init_db, get_session, JobPosting
    init_db(db_path)
    session = get_session(db_path)

    job = JobPosting(
        company="Acme",
        title="Software Engineer",
        description="We are hiring in a fast-paced environment. Apply now.",
    )
    session.add(job)
    session.commit()
    session.close()

    res_path = build_training_set(db_path=db_path, output_csv=csv_path)
    assert os.path.exists(res_path)

    import pandas as pd
    df = pd.read_csv(res_path)
    assert len(df) == 1
    assert "genericness_score" in df.columns
    assert "bootstrap_ghost_label" in df.columns
