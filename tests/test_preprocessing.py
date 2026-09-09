from datetime import datetime, timedelta
import pytest

from preprocessing.clean import clean_job_text, extract_metadata


def test_clean_job_text_messy_html():
    """Test text cleaning with messy HTML fixtures (script tags, inline CSS, HTML entities, unicode artifacts)."""
    messy_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>body { color: red; font-size: 14px; }</style>
        <script type="text/javascript">
            var tracker = { id: 12345, event: "impression" };
            function track() { console.log("tracking"); }
        </script>
    </head>
    <body>
        <nav><a href="/home">Home</a> | <a href="/jobs">Jobs</a></nav>
        <div class="job-container">
            <h1>Senior Python Engineer&nbsp;&amp;&nbsp;AI Lead</h1>
            <p class="summary">We are building state-of-the-art LLM pipelines\xa0at Acme Corp.</p>
            <div class="content">
                <span>Requirements:</span>
                <ul>
                    <li>5+ years of experience with Python, PyTorch &amp; FastAPI.</li>
                    <li>Strong understanding of distributed systems.</li>
                </ul>
            </div>
            <p>Equal Opportunity Employer. All qualified applicants will receive consideration. Apply Now</p>
        </div>
    </body>
    </html>
    """

    cleaned = clean_job_text(messy_html)

    # 1. Assert script & style contents are completely removed
    assert "var tracker" not in cleaned
    assert "console.log" not in cleaned
    assert "color: red" not in cleaned

    # 2. Assert HTML entities and unicode non-breaking spaces are unescaped/normalized
    assert "Senior Python Engineer & AI Lead" in cleaned
    assert "LLM pipelines at Acme Corp." in cleaned
    assert "PyTorch & FastAPI" in cleaned

    # 3. Assert casing is PRESERVED (not lowercased) for POS tagging
    assert "Senior Python Engineer" in cleaned
    assert "PyTorch" in cleaned
    assert "FastAPI" in cleaned

    # 4. Assert boilerplate is stripped
    assert "Equal Opportunity Employer" not in cleaned
    assert "Apply Now" not in cleaned


def test_clean_job_text_sentence_boundary_retention():
    """Verify sentence boundaries and punctuation are preserved for spaCy/BERT POS tagging."""
    raw_text = "We are hiring a Lead Architect.The candidate must master Kubernetes!Do you have experience?"
    cleaned = clean_job_text(raw_text)

    assert "Lead Architect. The candidate" in cleaned
    assert "Kubernetes! Do you" in cleaned
    assert cleaned.startswith("We are hiring")


def test_extract_metadata_dict():
    """Test extract_metadata with dictionary input."""
    posted_date = datetime.utcnow() - timedelta(days=5)
    posting_dict = {
        "company": "Anthropic",
        "title": "Member of Technical Staff",
        "salary_listed": "$180,000 - $240,000",
        "posted_date": posted_date,
    }

    meta = extract_metadata(posting_dict)

    assert meta["company"] == "Anthropic"
    assert meta["title"] == "Member of Technical Staff"
    assert meta["salary_listed"] is True
    assert meta["days_since_posted"] == 5


def test_extract_metadata_no_salary():
    """Test extract_metadata when salary is missing or None."""
    posting_dict = {
        "company": "Stripe",
        "title": "Backend Engineer",
        "salary_listed": None,
        "posted_date": None,
    }

    meta = extract_metadata(posting_dict)

    assert meta["company"] == "Stripe"
    assert meta["salary_listed"] is False
    assert meta["days_since_posted"] is None
