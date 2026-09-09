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
│   ├── clean.py
│   └── run.py
├── nlp/               # Ghost-signal feature extraction & Transformer/BERT classifier
│   ├── classifier.py
│   ├── evaluate.py
│   └── features.py
├── sentiment/         # VADER sentiment analysis & custom frustration lexicon
│   ├── analyze.py
│   └── run.py
├── scoring/           # Multi-factor scoring engine combining NLP & sentiment metrics
│   ├── final_score.py
│   └── run.py
├── seo/               # Static transparency site generator & Jinja2 templates
│   ├── build/
│   ├── templates/
│   └── build.py
├── lexicons/          # Phrase lexicons and weighted sentiment CSVs
├── dashboard/         # Streamlit web application & interactive visualizations
├── analytics/         # GA4 tracking and telemetry integration
├── data/              # SQLite database storage & raw/processed data folders (gitignored)
│   ├── raw/
│   └── processed/
├── tests/             # Automated test suite & HTML fixtures
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

### 6. Pipeline Execution Sequence
```bash
# 1. Initialize Database & Run Job Board Scraper
python -c "from scraper import init_db; init_db()"
python scraper/run.py --config config/target_companies.yaml

# 2. Run Review Scraper (PRAW Reddit)
python scraper/run_reviews.py --config config/target_companies.yaml

# 3. Clean Text & Extract Metadata
python preprocessing/run.py

# 4. Run Sentiment Analysis
python sentiment/run.py

# 5. Compute Final Ghost Scores & Correlation Check
python scoring/run.py

# 6. Generate Static SEO Transparency Site
python seo/build.py --threshold 60
```

### 7. Run Test Suite
Validate all 28 unit tests across project test suites:
```bash
pytest tests/
```

## Static Site Deployment Guide (GitHub Pages / Netlify / Vercel)

The static transparency site generated in `seo/build/` is completely self-contained HTML/CSS.

### Default Deployment Option: GitHub Pages (Recommended)
GitHub Pages provides free, zero-config static hosting directly inside this repository:
1. In your GitHub Repository, navigate to **Settings** -> **Pages**.
2. Under **Build and deployment** -> **Source**, select **Deploy from a branch**.
3. Choose the `main` (or `gh-pages`) branch and set the folder to `/seo/build` (or set up a GitHub Action to deploy `seo/build/`).
4. Save settings. Your site will be published live at `https://<user>.github.io/ghost-job-detector/`.

### Alternative Deployment Options
- **Netlify**: Connect your GitHub repository, set the Publish Directory to `seo/build/`, and set the build command to `python seo/build.py`.
- **Vercel**: Import your repository, select **Other** project type, and set the Output Directory to `seo/build/`.
- **Cloudflare Pages**: Direct upload or connect Git repository with build output `seo/build/`.

## Legal & Ethical Notes

1. **Public Data Scope Only**: The data collection pipeline strictly scrapes publicly available job posting metadata (job titles, descriptions, salary ranges, and posting dates) published on corporate career pages and public job indices. No private or password-protected content is accessed.
2. **Programmatic `robots.txt` Compliance**: All automated scraping requests check target site `robots.txt` rules using `urllib.robotparser.RobotFileParser`. Any URL path explicitly disallowed by site directives is automatically skipped.
3. **Rate Limiting & Server Politeness**: Scrapers implement random delays (2–5 seconds) between network requests and use realistic, transparent User-Agent headers to ensure minimal load on host web servers.
4. **No Storage of Candidate Data**: Ghost Job Detector is purely focused on corporate job postings and institutional transparency. The application does not collect, track, process, or store any personal candidate data, applicant resumes, or individual user identities.
