# Ghost Job Detector

**Ghost Job Detector** is an end-to-end data analytics, NLP, and machine learning system designed to identify non-viable, phantom, or stagnant job listings ("ghost jobs") across public job boards and community forums. 

By combining web scraping, text normalization, candidate sentiment lexicons, fine-tuned language models (BERT/Transformers), and multi-factor signal scoring, the system computes corporate transparency scores ($0 - 100$) and exposes insights via an interactive **Streamlit Dashboard** and static **SEO Transparency Site**.

---

## 📁 Directory Structure

```text
ghost-job-detector/
├── config/            # Target company taxonomy & scraper settings
│   └── target_companies.yaml
├── scraper/           # Playwright/BeautifulSoup scrapers & SQLAlchemy ORM
│   ├── jobboards.py
│   ├── reviews.py
│   ├── models.py
│   └── run.py
├── preprocessing/     # HTML text cleaning, normalization, metadata extraction
│   ├── clean.py
│   └── run.py
├── nlp/               # Ghost-signal features & BERT classifier
│   ├── classifier.py
│   ├── evaluate.py
│   └── features.py
├── sentiment/         # VADER sentiment analysis & candidate frustration lexicon
│   ├── analyze.py
│   └── run.py
├── scoring/           # Composite 0-100 ghost scoring engine & correlation engine
│   ├── final_score.py
│   └── run.py
├── seo/               # Static SEO transparency site generator & Jinja2 templates
│   ├── build/
│   ├── templates/
│   └── build.py
├── analytics/         # GA4 Reporting Data API client & PageAnalytics cache
│   └── ga4_client.py
├── dashboard/         # Streamlit 4-tab interactive web application
│   └── app.py
├── scripts/           # IEEE paper research artifact exporter
│   └── export_for_paper.py
├── reports/paper/     # Exported metrics CSVs, PNG scatter plots, and IEEE summary
├── lexicons/          # Weighted sentiment phrase lexicons
├── data/              # SQLite database storage (gitignored)
├── tests/             # Automated test suite (44 unit tests)
├── .github/workflows/ # Scheduled scraping cron jobs
├── README.md          # System documentation & execution guide
└── requirements.txt   # Python project dependencies
```

---

## 🚀 Quick Start & Execution Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/GarimaDixit2502/ghost-job-detector.git
cd ghost-job-detector

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browser engines & spaCy NLP model
playwright install chromium
python -m spacy download en_core_web_sm
```

### 2. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```

---

### 3. Running the Interactive Streamlit Dashboard 📊
To launch the 4-tab interactive web application:

```bash
streamlit run dashboard/app.py
```
> Navigate to **`http://localhost:8501`** in your browser.
> 
> **Dashboard Features**:
> - **Tab 1: Overview & Distribution** — KPI cards, score histogram, highest/lowest risk postings table.
> - **Tab 2: Industry Heatmap** — Industry taxonomy severity breakdown.
> - **Tab 3: Sentiment Correlation** — Candidate frustration vs. ghost score scatter plot with live Pearson $r$ / Spearman $\rho$.
> - **Tab 4: GA4 Traffic Telemetry** — Pageviews, average duration, bounce rate, and pilot validation telemetry.

---

### 4. Running the Complete End-to-End Pipeline ⚡

```bash
# 1. Scrape target company career pages & job boards
python scraper/run.py --config config/target_companies.yaml

# 2. Scrape candidate reviews (Reddit / Glassdoor stub)
python scraper/run_reviews.py --config config/target_companies.yaml

# 3. Clean HTML boilerplate & extract structured metadata
python preprocessing/run.py

# 4. Run candidate frustration sentiment analysis
python sentiment/run.py

# 5. Compute composite 0-100 Ghost Job Scores & model correlation check
python scoring/run.py

# 6. Generate static SEO transparency report pages (output to seo/build/)
python seo/build.py --threshold 60

# 7. Export IEEE paper research artifacts & plots (output to reports/paper/)
python scripts/export_for_paper.py
```

---

### 5. Running Automated Unit Tests 🧪
Validate all 44 test suites across scraping, NLP, sentiment, scoring, SEO, GA4, and export modules:

```bash
pytest tests/
```

---

## 🌐 Static Site Deployment Guide (GitHub Pages / Netlify / Vercel)

The transparency site generated in `seo/build/` is static HTML/CSS with JSON-LD structured data.

### Deploying to GitHub Pages (Recommended)
1. Navigate to **Settings** -> **Pages** in your GitHub repository (`GarimaDixit2502/ghost-job-detector`).
2. Under **Source**, choose **Deploy from a branch**.
3. Select the `main` branch and folder `/seo/build`.
4. Click **Save**. Your site will be published at `https://GarimaDixit2502.github.io/ghost-job-detector/`.

---

## 📊 IEEE Paper Research Artifact Exporter

Running `python scripts/export_for_paper.py` generates publication-ready artifacts in `reports/paper/`:
- **`classifier_metrics.csv`**: Precision, Recall, F1 score, Accuracy.
- **`confusion_matrix.png`**: High-resolution confusion matrix plot.
- **`sentiment_vs_ghost_score.png`**: Candidate frustration vs. Ghost Score scatter plot annotated with Pearson $r$.
- **`ga4_traffic_vs_ghost_score.png`**: 30-day GA4 pageviews vs. Ghost Score scatter plot annotated with Pearson $r$.
- **`summary_stats.md`**: Statistical summary table ready for IEEE paper Results section.

---

## ⚖️ Legal & Ethical Policy

1. **Public Data Scope Only**: The pipeline exclusively accesses public job posting metadata and public community discussion forums. No private or password-protected content is accessed.
2. **Programmatic `robots.txt` Compliance**: All scraping requests verify target site `robots.txt` directives using `urllib.robotparser.RobotFileParser`.
3. **Rate Limiting & Server Politeness**: Requests incorporate 2–5 second delays and transparent User-Agent headers.
4. **No Candidate Tracking**: Ghost Job Detector measures institutional posting behavior. No candidate personal data, applicant resumes, or individual identities are collected or stored.
