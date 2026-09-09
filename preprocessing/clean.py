import html
import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, Union

from bs4 import BeautifulSoup


BOILERPLATE_PATTERNS = [
    r"(?i)equal\s+opportunity\s+employer",
    r"(?i)all\s+qualified\s+applicants\s+will\s+receive\s+consideration",
    r"(?i)click\s+here\s+to\s+apply",
    r"(?i)apply\s+now\b",
    r"(?i)all\s+rights\s+reserved\.?",
    r"(?i)privacy\s+policy",
    r"(?i)terms\s+of\s+service",
]


def clean_job_text(raw_html_or_text: str) -> str:
    """Strips HTML tags, scripts, styles, boilerplate, and unicode artifacts from job text.

    Preserves letter casing and sentence boundaries required for downstream POS tagging and BERT embeddings.
    """
    if not raw_html_or_text or not isinstance(raw_html_or_text, str):
        return ""

    # Parse HTML with BeautifulSoup if markup tags exist
    if "<" in raw_html_or_text and ">" in raw_html_or_text:
        soup = BeautifulSoup(raw_html_or_text, "html.parser")
        # Remove non-content tags
        for el in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "svg"]):
            el.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = raw_html_or_text

    # Unescape HTML entities (&nbsp;, &amp;, &quot;, &lt;, &gt;)
    text = html.unescape(text)

    # Normalize unicode characters (converting \xa0, \u200b to standard spaces)
    text = unicodedata.normalize("NFKC", text)

    # Strip common non-essential boilerplate phrases
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)

    # Replace controls and non-printable characters (except normal punctuation and newlines)
    text = re.sub(r"[\r\t\f\v]+", " ", text)

    # Collapse multiple consecutive blank spaces while preserving single spaces and sentence punctuation (. ! ?)
    text = re.sub(r"[ \t]+", " ", text)

    # Ensure space after sentence-ending punctuation if missing (e.g., "Engineer.Apply" -> "Engineer. Apply")
    text = re.sub(r"([a-z0-9\.\)\'\"])([\.\!\?])([A-Z])", r"\1\2 \3", text)

    # Trim leading/trailing whitespace per line and overall
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned_text = "\n".join(lines)

    return cleaned_text.strip()


def extract_metadata(posting: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """Extract structured metadata fields from a posting dict or ORM JobPosting object.

    Does not re-parse unstructured text.
    """
    if isinstance(posting, dict):
        company = posting.get("company", "Unknown")
        title = posting.get("title", "Untitled")
        raw_salary = posting.get("salary_listed")
        posted_date = posting.get("posted_date")
        scraped_at = posting.get("scraped_at")
    else:
        company = getattr(posting, "company", "Unknown")
        title = getattr(posting, "title", "Untitled")
        raw_salary = getattr(posting, "salary_listed", None)
        posted_date = getattr(posting, "posted_date", None)
        scraped_at = getattr(posting, "scraped_at", None)

    # Evaluate salary_listed boolean indicator
    salary_listed = bool(raw_salary and str(raw_salary).strip().lower() not in ["none", "null", ""])

    # Calculate days_since_posted
    days_since_posted = None
    ref_time = datetime.utcnow()

    if posted_date and isinstance(posted_date, datetime):
        delta = ref_time - posted_date
        days_since_posted = max(0, delta.days)
    elif scraped_at and isinstance(scraped_at, datetime):
        delta = ref_time - scraped_at
        days_since_posted = max(0, delta.days)

    return {
        "company": company,
        "title": title,
        "salary_listed": salary_listed,
        "posting_date": posted_date,
        "days_since_posted": days_since_posted,
    }
