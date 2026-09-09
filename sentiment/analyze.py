import csv
import math
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session

from scraper.models import CompanyReview, init_db, get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy global SentimentIntensityAnalyzer instance
_VADER_ANALYZER: Optional[SentimentIntensityAnalyzer] = None


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Lazy initialization of NLTK VADER SentimentIntensityAnalyzer."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        try:
            nltk.download("vader_lexicon", quiet=True)
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK VADER analyzer: {e}")
            _VADER_ANALYZER = None
    return _VADER_ANALYZER


def load_frustration_lexicon(csv_path: Optional[str] = None) -> List[Tuple[str, float]]:
    """Load weighted hiring frustration phrases from CSV file."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "lexicons", "hiring_frustration_phrases.csv")

    phrases: List[Tuple[str, float]] = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    phrase = row.get("phrase", "").strip().lower()
                    try:
                        weight = float(row.get("weight", 0.5))
                    except ValueError:
                        weight = 0.5
                    if phrase:
                        phrases.append((phrase, weight))
            return phrases
        except Exception as e:
            logger.error(f"Error reading frustration lexicon CSV {csv_path}: {e}")

    # Fallback default lexicon if CSV reading fails
    return [
        ("never heard back", 0.85),
        ("ghosted after", 0.95),
        ("interviewed for a job that didn't exist", 1.00),
        ("position was already filled", 0.90),
        ("no response after final round", 0.95),
        ("fake job posting", 1.00),
        ("waste of time", 0.80),
        ("radio silence", 0.85),
    ]


def baseline_sentiment(text: str) -> float:
    """Calculate NLTK VADER baseline negativity score normalized to [0.0, 1.0].

    Returns compound negativity where 0.0 represents positive/neutral text and 1.0 represents highly negative text.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    analyzer = get_vader_analyzer()
    if analyzer is None:
        return 0.0

    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    neg = scores.get("neg", 0.0)

    # Convert compound [-1.0, 1.0] to negativity [0.0, 1.0] and blend with negative polarity ratio
    compound_negativity = (1.0 - compound) / 2.0
    blended_negativity = (0.7 * compound_negativity) + (0.3 * neg)

    return min(1.0, max(0.0, round(float(blended_negativity), 4)))


def frustration_score(text: str, lexicon_path: Optional[str] = None) -> float:
    """Calculate hiring frustration score (0.0 to 1.0) combining NLTK VADER baseline and weighted phrase match.

    ================================================================================
    FRUSTRATION SCORE COMBINATION FORMULA:
    --------------------------------------------------------------------------------
    frustration_score = 0.40 * vader_negativity + 0.60 * phrase_match_density

    - vader_negativity: NLTK VADER compound negativity score normalized to [0, 1].
    - phrase_match_density: Sum of matched phrase severity weights from
      'lexicons/hiring_frustration_phrases.csv', scaled to [0, 1].
    ================================================================================
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    lower_text = text.lower()
    vader_neg = baseline_sentiment(text)

    lexicon = load_frustration_lexicon(lexicon_path)
    matched_weight_sum = 0.0
    matched_count = 0

    for phrase, weight in lexicon:
        if phrase in lower_text:
            matched_weight_sum += weight
            matched_count += 1

    if matched_count == 0:
        phrase_density = 0.0
    else:
        # Scale matched weights (capped at 1.0)
        phrase_density = min(1.0, (matched_weight_sum / min(3.0, float(matched_count))) * 0.85)

    final_score = (0.40 * vader_neg) + (0.60 * phrase_density)
    return min(1.0, max(0.0, round(float(final_score), 4)))


def company_sentiment_score(
    company: str,
    db_path: str = "data/ghostjobs.db",
    decay_lambda: float = 0.03,
) -> Dict[str, Any]:
    """Calculate company-level aggregate frustration score using recency-weighted mean.

    ================================================================================
    RECENCY-WEIGHTED MEAN RATIONALE:
    --------------------------------------------------------------------------------
    Why Recency-Weighted Mean?
    Candidate reviews posted recently reflect active hiring practices, active ghost
    job postings, and current recruiter behavior far more accurately than legacy
    reviews from prior years. Each review 'i' is assigned an exponential decay weight:
        w_i = exp(-lambda * days_ago)   (default lambda = 0.03)
    The aggregated company frustration score is computed as:
        weighted_mean = sum(w_i * frustration_score_i) / sum(w_i)
    ================================================================================
    """
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        reviews = session.query(CompanyReview).filter_by(company=company).all()
        if not reviews:
            return {"company": company, "score": 0.0, "review_count": 0}

        now = datetime.utcnow()
        total_weight = 0.0
        weighted_score_sum = 0.0

        for r in reviews:
            posted_date = r.posted_at or r.scraped_at or now
            days_ago = max(0, (now - posted_date).days)
            weight = math.exp(-decay_lambda * days_ago)

            score_i = frustration_score(r.text)
            weighted_score_sum += weight * score_i
            total_weight += weight

        if total_weight == 0.0:
            agg_score = 0.0
        else:
            agg_score = weighted_score_sum / total_weight

        return {
            "company": company,
            "score": min(1.0, max(0.0, round(float(agg_score), 4))),
            "review_count": len(reviews),
        }
    finally:
        session.close()
