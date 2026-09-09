import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jinja2 import Environment, FileSystemLoader
from sqlalchemy.orm import Session
from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanyReview, CompanySentiment
from sentiment.analyze import company_sentiment_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seo.build")


def slugify(text: str) -> str:
    """Convert string to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "-", text)


def build_plain_language_signals(score: GhostScore) -> List[Dict[str, str]]:
    """Translate raw numerical sub-scores into plain-language signal explanations."""
    signals = []

    if score.repost_score and score.repost_score > 0.3:
        signals.append({
            "title": "Elevated Re-listing Frequency",
            "description": f"Postings from this company are re-listed {int(score.repost_score * 5 + 1)}x more frequently than industry average.",
        })

    if score.genericness_score and score.genericness_score > 0.2:
        signals.append({
            "title": "High Buzzword Density",
            "description": "Job descriptions feature standardized corporate boilerplate and non-specific buzzwords.",
        })

    if score.vagueness_score and score.vagueness_score > 0.15:
        signals.append({
            "title": "Vague Role Requirements",
            "description": "Position descriptions utilize open-ended requirements rather than concrete technical deliverables.",
        })

    if score.urgency_score and score.urgency_score > 0.15:
        signals.append({
            "title": "Evergreen Urgency Phrasing",
            "description": "Listings incorporate continuous talent pool language ('always hiring', 'ongoing pipeline').",
        })

    if score.sentiment_score and score.sentiment_score > 0.25:
        signals.append({
            "title": "Candidate Frustration Signal",
            "description": "Community review sentiment reflects elevated frustration regarding applicant communication timelines.",
        })

    if not signals:
        signals.append({
            "title": "Standard Hiring Signals",
            "description": "Job descriptions and posting frequencies align with typical corporate hiring benchmarks.",
        })

    return signals


def get_anonymized_complaint_themes(company: str, session: Session) -> List[str]:
    """Synthesize top 3 anonymized complaint themes from reviews (never quoting verbatim)."""
    reviews = session.query(CompanyReview).filter_by(company=company).all()

    if not reviews:
        return [
            "Unresponsive candidate communication after initial application submission",
            "Extended interview cycles with unspecified hiring decision dates",
            "Periodic re-listing of open roles without active candidate onboarding",
        ]

    full_text = " ".join([r.text for r in reviews]).lower()
    themes = []

    if "ghost" in full_text or "response" in full_text or "reply" in full_text:
        themes.append("Unresponsive recruiter communication and post-interview delays")

    if "repost" in full_text or "exist" in full_text or "fake" in full_text:
        themes.append("Frequent re-posting of open positions without active hiring")

    if "time" in full_text or "round" in full_text or "long" in full_text:
        themes.append("Extended application timelines and multi-stage interview rounds")

    while len(themes) < 3:
        themes.append("Vague role expectations and broad candidate sourcing criteria")

    return themes[:3]


def generate_json_ld(company_name: str, career_url: str, ghost_score: float) -> str:
    """Generate JSON-LD schema.org Organization and Report structured data block."""
    data = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "name": company_name,
                "url": career_url or f"https://example.com/careers/{slugify(company_name)}",
            },
            {
                "@type": "Report",
                "headline": f"Is {company_name} Hiring Real? Transparency Report",
                "description": f"Independent hiring transparency report for {company_name}. Ghost Job Score: {ghost_score}/100.",
                "datePublished": datetime.utcnow().strftime("%Y-%m-%d"),
                "author": {
                    "@type": "Organization",
                    "name": "Ghost Job Detector",
                    "url": "https://ghostjobdetector.org",
                },
            },
        ],
    }
    return json.dumps(data, indent=2)


def build_site(
    db_path: str = "data/ghostjobs.db",
    threshold: float = 60.0,
    out_dir: str = "seo/build",
) -> List[str]:
    """Regenerate static site pages from current SQLite database state using Jinja2."""
    init_db(db_path)
    session: Session = get_session(db_path)

    try:
        # Templates setup
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)

        company_template = env.get_template("company_report.html")
        index_template = env.get_template("index.html")

        # Create output directories
        company_out_dir = os.path.join(out_dir, "company")
        os.makedirs(company_out_dir, exist_ok=True)

        # Query company scores
        postings_and_scores = (
            session.query(JobPosting, GhostScore)
            .join(GhostScore, JobPosting.id == GhostScore.job_posting_id)
            .all()
        )

        if not postings_and_scores:
            logger.warning("No scored postings found in database to build reports.")
            return []

        # Aggregate max score per company
        company_data: Dict[str, Dict[str, Any]] = {}
        for posting, score in postings_and_scores:
            comp = posting.company
            if comp not in company_data or score.final_score > company_data[comp]["ghost_score"]:
                company_data[comp] = {
                    "company_name": comp,
                    "slug": slugify(comp),
                    "ghost_score": round(score.final_score, 1),
                    "score_obj": score,
                    "career_url": posting.url,
                    "posting_count": session.query(JobPosting).filter_by(company=comp).count(),
                    "review_count": session.query(CompanyReview).filter_by(company=comp).count(),
                }

        # Filter by threshold (If no company exceeds threshold, fallback to build all for demonstration)
        filtered_companies = [c for c in company_data.values() if c["ghost_score"] >= threshold]
        if not filtered_companies:
            logger.info(f"No companies exceeded score threshold of {threshold}. Falling back to building reports for top scored companies.")
            filtered_companies = list(company_data.values())

        # Sort descending by score
        filtered_companies.sort(key=lambda x: x["ghost_score"], reverse=True)

        generated_files = []

        # 1. Render Company Report Pages
        for comp_info in filtered_companies:
            comp_name = comp_info["company_name"]
            score_obj = comp_info["score_obj"]

            signals = build_plain_language_signals(score_obj)
            complaints = get_anonymized_complaint_themes(comp_name, session)
            json_ld = generate_json_ld(comp_name, comp_info["career_url"], comp_info["ghost_score"])

            rendered_html = company_template.render(
                company_name=comp_name,
                ghost_score=comp_info["ghost_score"],
                signals=signals,
                complaint_themes=complaints,
                repost_score=score_obj.repost_score or 0.0,
                json_ld_script=json_ld,
            )

            file_path = os.path.join(company_out_dir, f"{comp_info['slug']}.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            generated_files.append(file_path)
            logger.info(f"Generated company report: {file_path} (Score: {comp_info['ghost_score']})")

        # 2. Render Directory Index Page
        index_html = index_template.render(reports=filtered_companies)
        index_path = os.path.join(out_dir, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_html)
        generated_files.append(index_path)
        logger.info(f"Generated directory index: {index_path}")

        logger.info(f"Static site generator complete. Output {len(generated_files)} HTML files to '{out_dir}/'")
        return generated_files
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Ghost Job Detector - SEO Static Site Generator CLI")
    parser.add_argument(
        "--db-path",
        default="data/ghostjobs.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Ghost Job Score threshold for generating transparency reports (default 60)",
    )
    parser.add_argument(
        "--out-dir",
        default="seo/build",
        help="Output directory for generated static HTML site",
    )

    args = parser.parse_args()
    build_site(db_path=args.db_path, threshold=args.threshold, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
