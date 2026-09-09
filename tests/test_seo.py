import os
import json
import pytest
from bs4 import BeautifulSoup

from seo.build import build_site, generate_json_ld, slugify
from scraper.models import init_db, get_session, JobPosting, GhostScore


def test_slugify():
    """Verify URL slugification helper."""
    assert slugify("Anthropic AI") == "anthropic-ai"
    assert slugify("OpenAI, Inc.") == "openai-inc"


def test_generate_json_ld():
    """Verify schema.org JSON-LD generation for SEO structured data."""
    json_str = generate_json_ld("Acme Corp", "https://example.com/careers", 75.5)
    data = json.loads(json_str)

    assert "@context" in data
    assert "@graph" in data
    graph = data["@graph"]
    assert len(graph) == 2
    assert graph[0]["@type"] == "Organization"
    assert graph[0]["name"] == "Acme Corp"
    assert graph[1]["@type"] == "Report"
    assert "75.5/100" in graph[1]["description"]


def test_seo_build_generator_and_title_format(tmp_path):
    """Test full static site generation and verify SEO title format & JSON-LD inclusion."""
    db_file = tmp_path / "test_seo.db"
    db_path = str(db_file)
    out_dir = str(tmp_path / "build")

    init_db(db_path)
    session = get_session(db_path)

    company = "TechInnovators"
    job = JobPosting(
        company=company,
        title="Senior Python Architect",
        description="Fast-paced environment.",
    )
    session.add(job)
    session.commit()

    score = GhostScore(
        job_posting_id=job.id,
        genericness_score=0.4,
        vagueness_score=0.3,
        repost_score=0.2,
        urgency_score=0.1,
        bert_score=0.2,
        sentiment_score=0.3,
        final_score=68.5,
    )
    session.add(score)
    session.commit()
    session.close()

    generated_files = build_site(db_path=db_path, threshold=50.0, out_dir=out_dir)
    assert len(generated_files) == 2  # 1 company report + 1 index.html

    index_file = os.path.join(out_dir, "index.html")
    company_file = os.path.join(out_dir, "company", "techinnovators.html")

    assert os.path.exists(index_file)
    assert os.path.exists(company_file)

    # Inspect company report HTML structure
    with open(company_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Assert Title tag format
    title_text = soup.title.string.strip()
    assert title_text == "Is TechInnovators Hiring Real? | Transparency Report"

    # 2. Assert Meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    assert meta_desc is not None
    assert "TechInnovators" in meta_desc["content"]
    assert "68.5/100" in meta_desc["content"]

    # 3. Assert JSON-LD script block
    json_ld_script = soup.find("script", attrs={"type": "application/ld+json"})
    assert json_ld_script is not None
    json_data = json.loads(json_ld_script.string)
    assert json_data["@graph"][0]["name"] == "TechInnovators"
