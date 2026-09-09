# Ghost Job Detector

Ghost Job Detector is an end-to-end data analytics and NLP system designed to identify non-viable, phantom, or stagnant job listings ("ghost jobs") across public job boards and community forums. By combining scraping, text cleaning, sentiment lexicons, language models, and heuristic ghost-signal scoring, the application calculates transparency scores for job postings and presents real-time analytics through an interactive Streamlit dashboard and static SEO reporting site.

## Directory Structure

```text
ghost-job-detector/
├── scraper/           # Job board, Glassdoor, and Reddit web scraping & ORM models
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
├── tests/             # Automated test suite
├── .github/
│   └── workflows/     # GitHub Actions workflow specifications for scheduled scraping
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
playwright install
python -m spacy download en_core_web_sm
```

### 5. Environment Configuration
Copy the `.env.example` file to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```

### 6. Initialize Database
Initialize the SQLite database schema (`data/ghostjobs.db`):
```bash
python -c "from scraper import init_db; init_db()"
```

### 7. Run Tests
Validate the scaffolding setup:
```bash
pytest tests/
```
