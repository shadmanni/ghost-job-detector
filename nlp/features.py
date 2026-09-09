import os
import re
import logging
from typing import Dict, Any, Union, Optional

import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy global spaCy instance
_SPACY_NLP = None


def get_spacy_nlp():
    """Lazy initialization of spaCy English model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            logger.warning("spaCy model 'en_core_web_sm' not found directly. Loading blank English model.")
            _SPACY_NLP = spacy.blank("en")
    return _SPACY_NLP


GENERIC_BUZZWORDS = [
    "fast-paced environment",
    "wear many hats",
    "dynamic team",
    "competitive salary",
    "growth opportunity",
    "self-starter",
    "results-driven",
    "detail-oriented",
    "team player",
    "hit the ground running",
    "fast paced",
    "passionate individual",
    "exciting opportunities",
    "work hard play hard",
    "think outside the box",
    "synergy",
    "flexible mindset",
    "fast-growing company",
    "great culture",
    "career growth",
    "unmatched opportunity",
    "market-competitive",
    "industry leading",
    "fast-growing",
    "dynamic environment",
    "competitive compensation",
    "fast growing",
    "fast pace",
    "multi-tasker",
    "high energy",
]

VAGUE_NOUNS = {
    "opportunity",
    "environment",
    "team",
    "role",
    "position",
    "candidate",
    "responsibilities",
    "requirements",
    "stuff",
    "things",
    "task",
    "tasks",
    "work",
    "duties",
    "people",
    "individual",
    "person",
    "culture",
    "value",
    "values",
    "experience",
    "background",
    "company",
    "organization",
    "chance",
    "ability",
    "skills",
}


def load_urgency_phrases(lexicon_path: Optional[str] = None) -> list[str]:
    """Load urgency phrases from file or return default fallback lexicon."""
    if lexicon_path is None:
        # Default path relative to project root or lexicons folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lexicon_path = os.path.join(base_dir, "lexicons", "urgency_phrases.txt")

    if os.path.exists(lexicon_path):
        with open(lexicon_path, "r", encoding="utf-8") as f:
            phrases = [line.strip().lower() for line in f if line.strip()]
            return phrases

    # Fallback lexicon
    return [
        "always hiring",
        "immediate start",
        "ongoing opportunities",
        "apply now",
        "pool of candidates",
        "continuous hiring",
        "future opportunities",
        "accepting applications on an ongoing basis",
        "talent pool",
        "future consideration",
        "rolling basis",
        "urgently hiring",
        "immediate opening",
        "open pipeline",
        "evergreen posting",
    ]


def genericness_score(text: str) -> float:
    """Calculate genericness score (0.0 to 1.0) based on corporate buzzword phrase density."""
    if not text or len(text.strip()) == 0:
        return 0.0

    lower_text = text.lower()
    words = re.findall(r"\b\w+\b", lower_text)
    total_words = len(words)
    if total_words == 0:
        return 0.0

    match_count = sum(1 for phrase in GENERIC_BUZZWORDS if phrase in lower_text)
    # Calculate density normalized by word length
    density = (match_count / max(15, total_words)) * 25.0
    return min(1.0, max(0.0, round(float(density), 4)))


def vagueness_score(text: str) -> float:
    """Calculate vagueness score (0.0 to 1.0) using spaCy POS tagging.

    Ratio of generic vague nouns vs total concrete nouns.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    nlp = get_spacy_nlp()
    doc = nlp(text)

    total_nouns = 0
    vague_nouns_count = 0

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            total_nouns += 1
            lemma = token.lemma_.lower()
            # If noun is in vague list and not a proper noun or modified by a number
            has_nummod = any(child.dep_ in ["nummod", "quantmod"] for child in token.children)
            if lemma in VAGUE_NOUNS and token.pos_ != "PROPN" and not has_nummod:
                vague_nouns_count += 1

    if total_nouns == 0:
        return 0.0

    score = vague_nouns_count / float(total_nouns)
    return min(1.0, max(0.0, round(float(score), 4)))


def repost_score(posting: Union[Dict[str, Any], Any]) -> float:
    """Calculate repost score (0.0 to 1.0) based on repost_of_id / repost counter."""
    if posting is None:
        return 0.0

    if isinstance(posting, dict):
        repost_of_id = posting.get("repost_of_id")
        repost_count = posting.get("repost_count", 1 if repost_of_id else 0)
    else:
        repost_of_id = getattr(posting, "repost_of_id", None)
        repost_count = getattr(posting, "repost_count", 1 if repost_of_id else 0)

    if not repost_of_id and repost_count == 0:
        return 0.0

    # Normalize repost count (1 repost = 0.5, 5+ = 1.0)
    score = min(1.0, 0.4 + (repost_count * 0.15))
    return min(1.0, max(0.0, round(float(score), 4)))


def urgency_score(text: str, lexicon_path: Optional[str] = None) -> float:
    """Calculate urgency score (0.0 to 1.0) based on evergreen/urgency phrase density."""
    if not text or len(text.strip()) == 0:
        return 0.0

    lower_text = text.lower()
    words = re.findall(r"\b\w+\b", lower_text)
    total_words = len(words)
    if total_words == 0:
        return 0.0

    urgency_phrases = load_urgency_phrases(lexicon_path)
    match_count = sum(1 for phrase in urgency_phrases if phrase in lower_text)

    density = (match_count / max(15, total_words)) * 30.0
    return min(1.0, max(0.0, round(float(density), 4)))
