# Ghost Job Detector

Ghost Job Detector is an end-to-end data analytics and NLP system designed to identify non-viable, phantom, or stagnant job listings ("ghost jobs") across public job boards and community forums. By combining scraping, text cleaning, sentiment lexicons, language models, and heuristic ghost-signal scoring, the application calculates transparency scores for job postings and presents real-time analytics through an interactive Streamlit dashboard and static SEO reporting site.

## Directory Structure

```text
ghost-job-detector/
├── config/            # Target company lists and scraper parameters
│   └── target_companies.yaml
├── scraper/           # Web scraping engine, robots.txt checker, and ORM models
│   ├── jobboards.py
│   ├── models.py
│   └── run.py
├── preprocessing/     # Text cleaning, normalization, and metadata extraction
├── nlp/               # Ghost-signal feature extraction & Transformer/BERT classifier
├── sentiment/         # VADER sentiment analysis & custom frustration lexicon
├── scoring/           # Multi-factor scoring engine combining NLP & sentiment metrics
├── seo/               # Static transparency website generator
├── dashboard/         # Streamlit web application & interactive visualizations
├── analytics/         # GA4 tracking and telemetry integration
├── data/              # SQLite database storage & raw/processed data folders (gitignored)
│   ├── raw/
│   └── processed/
├── tests/             # Automated test suite & HTML fixtures
│   ├── fixtures/
│   └── test_jobboards.py
├── .github/
│   └── workflows/     # Scheduled data collection workflows
├── README.md          # Project documentation and setup guide
├── requirements.txt   # Python project dependencies
├── .env.example       # Template for API keys and configuration settings
└── .gitignore         # Git ignore rules
```

## Setup Instructions

### 1. Prerequisites
Ensure Python 3.11 (or Python 3.9+) is installed on your system.

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Playwright Browsers & spaCy Language Model
```bash
playwright install chromium
python -m spacy download en_core_web_sm
```

### 5. Environment Configuration
Copy the `.env.example` file to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

### 6. Initialize Database & Run Scraper CLI
```bash
python -c "from scraper import init_db; init_db()"
python scraper/run.py --config config/target_companies.yaml
```

### 7. Run Tests
Validate the test suite using saved HTML fixtures:
```bash
pytest tests/
```

## Legal & Ethical Notes

1. **Public Data Scope Only**: The data collection pipeline strictly scrapes publicly available job posting metadata (job titles, descriptions, salary ranges, and posting dates) published on corporate career pages and public job indices. No private or password-protected content is accessed.
2. **Programmatic `robots.txt` Compliance**: All automated scraping requests check target site `robots.txt` rules using `urllib.robotparser.RobotFileParser`. Any URL path explicitly disallowed by site directives is automatically skipped.
3. **Rate Limiting & Server Politeness**: Scrapers implement random delays (2–5 seconds) between network requests and use realistic, transparent User-Agent headers to ensure minimal load on host web servers.
4. **No Storage of Candidate Data**: Ghost Job Detector is purely focused on corporate job postings and institutional transparency. The application does not collect, track, process, or store any personal candidate data, applicant resumes, or individual user identities.
