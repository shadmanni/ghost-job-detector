import os
import sys
from datetime import datetime, timedelta
import random

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scraper.models import init_db, get_session, CompanyReview

SAMPLE_REVIEWS = [
    # Databricks (Higher frustration, ghosting complaints)
    {
        "company": "Databricks",
        "source": "reddit",
        "text": "Applied for a solutions architect role at Databricks. Passed two technical screens then complete radio silence for 6 weeks. Total waste of time and fake job posting.",
        "days_ago": 4,
    },
    {
        "company": "Databricks",
        "source": "reddit",
        "text": "Databricks recruiter reached out on LinkedIn, scheduled a call, then never heard back. Saw the exact same job reposted 3 days later.",
        "days_ago": 12,
    },
    {
        "company": "Databricks",
        "source": "reddit",
        "text": "Did 4 interview rounds at Databricks only to be told the position was already filled internally. Why did you even interview external candidates?",
        "days_ago": 22,
    },
    {
        "company": "Databricks",
        "source": "reddit",
        "text": "Standard technical assessment at Databricks. Coding challenges were hard, communication was slow but eventually got an automated rejection.",
        "days_ago": 35,
    },

    # Airbnb (Moderate to elevated frustration, stagnant listings)
    {
        "company": "Airbnb",
        "source": "reddit",
        "text": "Airbnb has had the same product designer job listing open for 8 months. Applied twice, ghosted after recruiter screen both times.",
        "days_ago": 6,
    },
    {
        "company": "Airbnb",
        "source": "reddit",
        "text": "Interview experience with Airbnb was decent initially, but after the virtual onsite they completely ghosted me. No response to follow-up emails.",
        "days_ago": 15,
    },
    {
        "company": "Airbnb",
        "source": "reddit",
        "text": "Applied for backend role at Airbnb. Got a prompt rejection within a week. At least they didn't leave me hanging in a talent pipeline.",
        "days_ago": 28,
    },

    # OpenAI (Moderate frustration, high applicant volume)
    {
        "company": "OpenAI",
        "source": "reddit",
        "text": "Applied to OpenAI research engineer position. Very high bar. Took about a month to hear back, process was intense but recruiter was upfront about timeline.",
        "days_ago": 8,
    },
    {
        "company": "OpenAI",
        "source": "reddit",
        "text": "Ghosted after take-home project for OpenAI. Spent 15 hours on it. Super disappointing experience for a leading lab.",
        "days_ago": 20,
    },
    {
        "company": "OpenAI",
        "source": "reddit",
        "text": "The OpenAI interview was rigorous. Felt like a legitimate open headcount, well-organized scheduling with hiring manager.",
        "days_ago": 40,
    },

    # Anthropic (Low frustration, responsive recruiters)
    {
        "company": "Anthropic",
        "source": "reddit",
        "text": "Had a fantastic interview experience with Anthropic. Transparent timeline, respectful interviewers, and recruiter provided detailed updates every 3 days.",
        "days_ago": 5,
    },
    {
        "company": "Anthropic",
        "source": "reddit",
        "text": "Anthropic process was smooth and fast. Rejection was prompt and polite after system design round. No ghosting whatsoever.",
        "days_ago": 18,
    },
    {
        "company": "Anthropic",
        "source": "reddit",
        "text": "Very communicative team at Anthropic. Clear expectations for the technical evaluation and competitive compensation transparency.",
        "days_ago": 30,
    },

    # Stripe (Lowest frustration, high operational excellence)
    {
        "company": "Stripe",
        "source": "reddit",
        "text": "Stripe's interview process is gold standard. Practical coding environment, responsive coordinators, and timely feedback throughout all rounds.",
        "days_ago": 7,
    },
    {
        "company": "Stripe",
        "source": "reddit",
        "text": "Did not pass Stripe's final round, but recruiter called personally with constructive feedback within 48 hours. Zero ghosting.",
        "days_ago": 16,
    },
    {
        "company": "Stripe",
        "source": "reddit",
        "text": "Applied to Stripe software engineer role. Assessment link arrived immediately, clear instructions and polite email updates throughout.",
        "days_ago": 25,
    },
]


def seed_reviews(db_path: str = "data/ghostjobs.db"):
    """Seed sample candidate reviews into company_reviews table."""
    init_db(db_path)
    session = get_session(db_path)
    now = datetime.utcnow()

    try:
        # Clear existing reviews if any
        session.query(CompanyReview).delete()

        count = 0
        for r in SAMPLE_REVIEWS:
            posted_time = now - timedelta(days=r["days_ago"], hours=random.randint(1, 12))
            review = CompanyReview(
                company=r["company"],
                source=r["source"],
                text=r["text"],
                posted_at=posted_time,
                scraped_at=now,
            )
            session.add(review)
            count += 1

        session.commit()
        print(f"Successfully seeded {count} Reddit candidate reviews into '{db_path}'.")
        return count
    finally:
        session.close()


if __name__ == "__main__":
    db_file = "data/ghostjobs.db"
    seed_reviews(db_file)
