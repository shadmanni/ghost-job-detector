import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanySentiment
from nlp.features import (
    genericness_score,
    vagueness_score,
    repost_score,
    urgency_score,
)
from nlp.classifier import predict
from sentiment.analyze import company_sentiment_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Named constant dictionary defining sub-score weights (Must sum to 1.0 / 100%)
DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "genericness": 0.20,
    "vagueness": 0.20,
    "repost": 0.20,
    "urgency": 0.15,
    "bert_classifier": 0.15,
    "company_sentiment": 0.10,
}


def calculate_final_ghost_score(
    genericness: float,
    vagueness: float,
    repost: float,
    urgency: float,
    bert_score: float,
    sentiment_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate single 0-100 Ghost Job Score from 6 sub-scores using weighted sum.

    Sub-scores are expected in range [0.0, 1.0]. The final score is scaled to [0.0, 100.0].
    """
    w = weights if weights is not None else DEFAULT_SCORE_WEIGHTS
    weight_sum = sum(w.values())

    if weight_sum == 0:
        return 0.0

    raw_weighted_sum = (
        (w.get("genericness", 0.20) * genericness)
        + (w.get("vagueness", 0.20) * vagueness)
        + (w.get("repost", 0.20) * repost)
        + (w.get("urgency", 0.15) * urgency)
        + (w.get("bert_classifier", 0.15) * bert_score)
        + (w.get("company_sentiment", 0.10) * sentiment_score)
    )

    # Normalize by total weight and scale to 0-100
    normalized_sum = raw_weighted_sum / weight_sum
    final_score = normalized_sum * 100.0

    return min(100.0, max(0.0, round(float(final_score), 2)))


def compute_and_store_scores(
    db_path: str = "data/ghostjobs.db",
    weights: Optional[Dict[str, float]] = None,
) -> int:
    """Compute sub-scores and final 0-100 Ghost Job Score for every posting and write to GhostScore table."""
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        postings = session.query(JobPosting).all()
        logger.info(f"Computing ghost job scores for {len(postings)} JobPosting records in '{db_path}'...")

        if not postings:
            return 0

        processed = 0
        for posting in postings:
            text = posting.cleaned_text or posting.description or ""

            # 1. Sub-score calculations
            g_score = genericness_score(text)
            v_score = vagueness_score(text)
            r_score = repost_score(posting)
            u_score = urgency_score(text)
            b_score = predict(text, posting)

            # Query company sentiment score
            c_sent = company_sentiment_score(posting.company, db_path=db_path)
            s_score = c_sent.get("score", 0.0)

            # 2. Final composite ghost score
            final_ghost_score = calculate_final_ghost_score(
                genericness=g_score,
                vagueness=v_score,
                repost=r_score,
                urgency=u_score,
                bert_score=b_score,
                sentiment_score=s_score,
                weights=weights,
            )

            # 3. Upsert into GhostScore ORM table
            existing_score = session.query(GhostScore).filter_by(job_posting_id=posting.id).first()

            if existing_score:
                existing_score.genericness_score = g_score
                existing_score.vagueness_score = v_score
                existing_score.repost_score = r_score
                existing_score.urgency_score = u_score
                existing_score.bert_score = b_score
                existing_score.sentiment_score = s_score
                existing_score.final_score = final_ghost_score
                existing_score.computed_at = datetime.utcnow()
            else:
                new_ghost_score = GhostScore(
                    job_posting_id=posting.id,
                    genericness_score=g_score,
                    vagueness_score=v_score,
                    repost_score=r_score,
                    urgency_score=u_score,
                    bert_score=b_score,
                    sentiment_score=s_score,
                    final_score=final_ghost_score,
                    computed_at=datetime.utcnow(),
                )
                session.add(new_ghost_score)
            processed += 1

        session.commit()
        logger.info(f"Successfully computed and persisted GhostScore entries for {processed} postings.")
        return processed
    except Exception as e:
        session.rollback()
        logger.error(f"Error computing and storing ghost scores: {e}")
        raise e
    finally:
        session.close()


def correlation_check(db_path: str = "data/ghostjobs.db") -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation between classifier score (bert_score) and company sentiment score.

    Validates whether external candidate feedback validates internal model predictions.
    """
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        scores = session.query(GhostScore).all()
        if not scores or len(scores) < 2:
            logger.warning("Not enough GhostScore records to compute correlation (minimum 2 required).")
            return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_r": 0.0, "spearman_p": 1.0, "sample_size": len(scores)}

        bert_scores = [s.bert_score for s in scores if s.bert_score is not None and s.sentiment_score is not None]
        sentiment_scores = [s.sentiment_score for s in scores if s.bert_score is not None and s.sentiment_score is not None]

        if len(bert_scores) < 2 or len(set(bert_scores)) <= 1 or len(set(sentiment_scores)) <= 1:
            logger.info(f"Constant or uniform score values across sample size N={len(bert_scores)}. Returning default correlation.")
            return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_r": 0.0, "spearman_p": 1.0, "sample_size": len(bert_scores)}

        p_r, p_p = stats.pearsonr(bert_scores, sentiment_scores)
        s_r, s_p = stats.spearmanr(bert_scores, sentiment_scores)

        results = {
            "pearson_r": round(float(p_r), 4),
            "pearson_p": round(float(p_p), 4),
            "spearman_r": round(float(s_r), 4),
            "spearman_p": round(float(s_p), 4),
            "sample_size": len(bert_scores),
        }

        logger.info("==========================================================")
        logger.info("VALIDATION CHECK: INTERNAL MODEL VS EXTERNAL SENTIMENT")
        logger.info("----------------------------------------------------------")
        logger.info(f"  Sample Size (N): {results['sample_size']}")
        logger.info(f"  Pearson Correlation (r):  {results['pearson_r']:.4f} (p-value: {results['pearson_p']:.4f})")
        logger.info(f"  Spearman Correlation (ρ): {results['spearman_r']:.4f} (p-value: {results['spearman_p']:.4f})")
        logger.info("==========================================================")

        return results
    finally:
        session.close()
