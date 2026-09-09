import pytest
from scoring.final_score import (
    DEFAULT_SCORE_WEIGHTS,
    calculate_final_ghost_score,
    compute_and_store_scores,
    correlation_check,
)
from scraper.models import init_db, get_session, JobPosting, GhostScore


def test_weights_sum_to_one():
    """Verify that default sub-score weights dictionary sums to exactly 1.0 (100%)."""
    total_weight = sum(DEFAULT_SCORE_WEIGHTS.values())
    assert total_weight == pytest.approx(1.0)


def test_calculate_final_ghost_score_edge_cases():
    """Verify final score formula bounds for edge cases (all zeros, all ones, mixed values)."""
    # Edge case 1: All zeros -> 0.0
    score_min = calculate_final_ghost_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert score_min == 0.0

    # Edge case 2: All ones -> 100.0
    score_max = calculate_final_ghost_score(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert score_max == 100.0

    # Edge case 3: All 0.5 -> 50.0
    score_mid = calculate_final_ghost_score(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert score_mid == 50.0

    # Out of bounds safety check
    score_overflow = calculate_final_ghost_score(1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
    assert score_overflow == 100.0


def test_compute_and_store_scores(tmp_path):
    """Verify end-to-end score computation and GhostScore database persistence."""
    db_file = tmp_path / "test_scoring.db"
    db_path = str(db_file)
    init_db(db_path)

    session = get_session(db_path)
    job = JobPosting(
        company="Acme Corp",
        title="Software Engineer",
        description="Fast-paced environment with competitive salary. Apply now.",
    )
    session.add(job)
    session.commit()
    job_id = job.id
    session.close()

    count = compute_and_store_scores(db_path=db_path)
    assert count == 1

    session = get_session(db_path)
    g_score = session.query(GhostScore).filter_by(job_posting_id=job_id).first()
    assert g_score is not None
    assert g_score.genericness_score >= 0.0
    assert g_score.vagueness_score >= 0.0
    assert g_score.final_score >= 0.0 and g_score.final_score <= 100.0
    session.close()


def test_correlation_check_graceful(tmp_path):
    """Verify correlation check returns dictionary structure even with minimal data."""
    db_file = tmp_path / "test_corr.db"
    db_path = str(db_file)
    init_db(db_path)

    res = correlation_check(db_path=db_path)
    assert "pearson_r" in res
    assert "spearman_r" in res
    assert "sample_size" in res
