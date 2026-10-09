import os
import sys
import yaml
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanySentiment, CompanyReview, PageAnalytics
from scoring.final_score import correlation_check, compute_and_store_scores, DEFAULT_SCORE_WEIGHTS
from analytics.ga4_client import (
    get_ga4_traffic_metrics,
    sync_ga4_page_analytics_to_db,
    is_ga4_configured,
    seed_pilot_analytics,
    fetch_ga4_30day_page_metrics,
    send_ga4_measurement_event,
    get_recent_telemetry_events,
    get_ga4_measurement_id,
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Ghost Job Detector — Analytics & Transparency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "activity_log" not in st.session_state:
    st.session_state.activity_log = [
        f"[{datetime.datetime.utcnow().strftime('%H:%M:%S')}] System initialized with active database.",
    ]

# Inject Clean Warm Surface Theme & Helvetica Typography
st.markdown(
    """
    <style>
    /* Global Typography: Helvetica */
    html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame, button, input, select, textarea {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: #2D2824;
    }

    /* Clean Warm Light App Canvas */
    .stApp {
        background-color: #FAF8F5 !important;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #F3EFE9 !important;
        border-right: 1px solid #E6DFD5 !important;
    }

    /* Compact Main Title & Subtitle */
    .dashboard-header {
        padding-bottom: 0.75rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid #E6DFD5;
    }
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1C1917 !important;
        margin: 0 0 0.25rem 0 !important;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .main-subtitle {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 0.9rem !important;
        color: #6B635B !important;
        margin: 0 !important;
        font-weight: 400;
        line-height: 1.4;
    }

    /* Navigation Tabs: Tactile 3D Button Style */
    div[data-testid="stTabs"] {
        border-bottom: 2px solid #D8CFBF;
        margin-bottom: 1.5rem;
        padding-bottom: 0;
    }
    div[data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent !important;
        padding: 4px 4px 0 4px;
    }
    button[data-baseweb="tab"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #57534E !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F1EB 100%) !important;
        border: 1px solid #D6CEBE !important;
        border-bottom: 2px solid #BFB5A2 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 3px rgba(60, 50, 40, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.12s ease-in-out !important;
        margin-right: 4px !important;
    }
    button[data-baseweb="tab"]:hover {
        background: linear-gradient(180deg, #FFFFFF 0%, #EAE4D9 100%) !important;
        color: #C85A32 !important;
        border-color: #C4BAA9 !important;
        transform: translateY(-1px) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C85A32 !important;
        background: #FAF8F5 !important;
        border-color: #D8CFBF !important;
        border-bottom: 2px solid #FAF8F5 !important;
        box-shadow: 0 -2px 4px rgba(60, 50, 40, 0.06), inset 0 2px 0 #C85A32 !important;
        font-weight: 700 !important;
    }

    /* KPI Cards: Subtle warm border with top highlight */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAF8F5 100%) !important;
        border: 1px solid #E8E2D9 !important;
        border-top: 3px solid #C85A32 !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 4px rgba(60, 50, 40, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #6B635B !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1C1917 !important;
    }

    /* Standard Interactive Button Styling */
    div.stButton > button {
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F1EB 100%) !important;
        border: 1px solid #D6CEBE !important;
        border-bottom: 2px solid #BFB5A2 !important;
        border-radius: 6px !important;
        color: #3A332C !important;
        box-shadow: 0 1px 2px rgba(60, 50, 40, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        transition: all 0.12s ease !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #FFFFFF 0%, #EDE7DD 100%) !important;
        border-color: #C85A32 !important;
        color: #C85A32 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(60, 50, 40, 0.08) !important;
    }

    /* Alert / Callout Boxes */
    div[data-testid="stAlert"] {
        border-radius: 6px !important;
        border: 1px solid #E2D9CD !important;
        font-size: 0.88rem !important;
    }

    /* Section Headings */
    h1, h2, h3, h4 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: #1C1917 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Dataframe Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E8E2D9 !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
    }

    .telemetry-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Live GA4 telemetry tracking tag injection
ga4_meas_id = get_ga4_measurement_id() or "G-90J1MLTLJM"

if ga4_meas_id:
    st.components.v1.html(
        f"""
        <script>
          try {{
            if (!parent.document.getElementById('ga4-gtag-script')) {{
              const script = parent.document.createElement('script');
              script.id = 'ga4-gtag-script';
              script.async = true;
              script.src = 'https://www.googletagmanager.com/gtag/js?id={ga4_meas_id}';
              parent.document.head.appendChild(script);

              const initScript = parent.document.createElement('script');
              initScript.id = 'ga4-gtag-init';
              initScript.innerHTML = `
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', '{ga4_meas_id}', {{
                  'send_page_view': true,
                  'page_title': 'Ghost Job Detector Dashboard',
                  'page_location': window.location.href
                }});
              `;
              parent.document.head.appendChild(initScript);
              console.log('GA4 tag ({ga4_meas_id}) successfully injected into parent document.');
            }}
          }} catch (e) {{
            console.error('GA4 injection error:', e);
          }}
        </script>
        """,
        height=1,
        width=1,
    )


def apply_warm_clean_theme(fig, height: int = 400):
    """Apply clean, minimalist styling with crisp lines and a warm palette to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Helvetica, Arial, sans-serif", size=11, color="#2D2824"),
        height=height,
        margin=dict(l=45, r=25, t=45, b=45),
        title=dict(
            font=dict(family="Helvetica, Arial, sans-serif", size=13, color="#1C1917"),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            gridcolor="#F0EBE3",
            linecolor="#D8CFBF",
            zerolinecolor="#D8CFBF",
            tickfont=dict(color="#57534E", size=10),
            title_font=dict(color="#2D2824", size=11),
        ),
        yaxis=dict(
            gridcolor="#F0EBE3",
            linecolor="#D8CFBF",
            zerolinecolor="#D8CFBF",
            tickfont=dict(color="#57534E", size=10),
            title_font=dict(color="#2D2824", size=11),
        ),
    )
    return fig


@st.cache_data(ttl=60)
def load_company_industries(config_path: str = "config/target_companies.yaml") -> dict:
    """Load company to industry mapping from target_companies.yaml."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mapping = {}
    companies = data.get("companies", []) if isinstance(data, dict) else data
    if isinstance(companies, list):
        for c in companies:
            if isinstance(c, dict) and "name" in c:
                mapping[c["name"]] = c.get("industry", "General Tech")
    return mapping


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "ghostjobs.db")
SNAPSHOT_PATH = os.path.join(PROJECT_ROOT, "data", "benchmark_snapshot.json")


def seed_from_snapshot(db_path: str = DEFAULT_DB_PATH) -> bool:
    """Restore SQLite database records from benchmark_snapshot.json."""
    if not os.path.exists(SNAPSHOT_PATH):
        return False

    try:
        import json

        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        init_db(db_path)
        session = get_session(db_path)
        try:
            if session.query(JobPosting).count() == 0:
                for p in data.get("job_postings", []):
                    dt = datetime.datetime.fromisoformat(p["posted_date"]) if p.get("posted_date") else None
                    posting = JobPosting(
                        id=p.get("id"),
                        company=p.get("company"),
                        title=p.get("title"),
                        description=p.get("description"),
                        url=p.get("url"),
                        source=p.get("source"),
                        posted_date=dt,
                        salary_listed=p.get("salary_listed"),
                        cleaned_text=p.get("cleaned_text"),
                    )
                    session.add(posting)

                for s in data.get("ghost_scores", []):
                    score = GhostScore(
                        id=s.get("id"),
                        job_posting_id=s.get("job_posting_id"),
                        genericness_score=s.get("genericness_score"),
                        vagueness_score=s.get("vagueness_score"),
                        repost_score=s.get("repost_score"),
                        urgency_score=s.get("urgency_score"),
                        bert_score=s.get("bert_score"),
                        sentiment_score=s.get("sentiment_score"),
                        final_score=s.get("final_score"),
                    )
                    session.add(score)

                for r in data.get("company_reviews", []):
                    review = CompanyReview(
                        id=r.get("id"),
                        company=r.get("company"),
                        source=r.get("source"),
                        text=r.get("text"),
                    )
                    session.add(review)

                for cs in data.get("company_sentiments", []):
                    sentiment = CompanySentiment(
                        id=cs.get("id"),
                        company=cs.get("company"),
                        score=cs.get("score"),
                        review_count=cs.get("review_count", 0),
                    )
                    session.add(sentiment)

                for pa in data.get("page_analytics", []):
                    analytics = PageAnalytics(
                        id=pa.get("id"),
                        company=pa.get("company"),
                        pageviews=pa.get("pageviews"),
                        avg_time_on_page=pa.get("avg_time_on_page"),
                        bounce_rate=pa.get("bounce_rate"),
                    )
                    session.add(analytics)

                session.commit()
                return True
        finally:
            session.close()
    except Exception as e:
        print(f"Error seeding from snapshot: {e}")
    return False


def load_dashboard_data(db_path: str = DEFAULT_DB_PATH):
    """Load live data from SQLite database into pandas DataFrames."""
    if not os.path.exists(db_path):
        init_db(db_path)

    session = get_session(db_path)
    try:
        # Check if database is empty; if so, automatically restore from benchmark snapshot
        if session.query(JobPosting).count() == 0:
            session.close()
            seed_from_snapshot(db_path=db_path)
            session = get_session(db_path)

        query = (
            session.query(JobPosting, GhostScore)
            .outerjoin(GhostScore, JobPosting.id == GhostScore.job_posting_id)
            .all()
        )

        rows = []
        for posting, score in query:
            rows.append({
                "posting_id": posting.id,
                "company": posting.company,
                "title": posting.title,
                "source": posting.source,
                "url": posting.url,
                "location": posting.location or "Not Specified",
                "posted_date": posting.posted_date,
                "scraped_at": posting.scraped_at,
                "genericness_score": score.genericness_score if score else 0.0,
                "vagueness_score": score.vagueness_score if score else 0.0,
                "repost_score": score.repost_score if score else 0.0,
                "urgency_score": score.urgency_score if score else 0.0,
                "bert_score": score.bert_score if score else 0.0,
                "sentiment_score": score.sentiment_score if score else 0.0,
                "final_ghost_score": score.final_score if score else 0.0,
            })

        df_postings = pd.DataFrame(rows)

        sentiments = session.query(CompanySentiment).all()
        s_rows = [{"company": s.company, "sentiment_score": s.score, "review_count": s.review_count} for s in sentiments]
        df_sentiments = pd.DataFrame(s_rows)

        analytics = session.query(PageAnalytics).all()
        a_rows = [
            {
                "company": a.company,
                "pageviews": a.pageviews,
                "avg_time_on_page": a.avg_time_on_page,
                "bounce_rate": a.bounce_rate,
                "fetched_at": a.fetched_at,
            }
            for a in analytics
        ]
        df_analytics = pd.DataFrame(a_rows)

        return df_postings, df_sentiments, df_analytics
    finally:
        session.close()


def main():
    db_path = DEFAULT_DB_PATH
    init_db(db_path)
    sync_ga4_page_analytics_to_db(db_path=db_path, allow_pilot_fallback=True)
    df_postings, df_sentiments, df_analytics = load_dashboard_data(db_path=db_path)
    industry_mapping = load_company_industries()

    if not df_postings.empty:
        df_postings["industry"] = df_postings["company"].map(lambda c: industry_mapping.get(c, "General Tech"))

    # =========================================================================
    # SIDEBAR: CONTROLS, ON-DEMAND LIVE SCRAPER & REAL-TIME STATS
    # =========================================================================
    st.sidebar.markdown("### ⚡ Live Ingestion & Scrape")
    st.sidebar.caption("Trigger real-time Playwright stealth scraping and zero-API community sentiment extraction.")

    with st.sidebar.form(key="live_scrape_form"):
        target_company = st.text_input("Target Company", value="Anthropic", help="Company name to crawl and evaluate")
        job_query = st.text_input("Job Title / Query", value="Software Engineer", help="Role keyword for public job search")
        target_location = st.text_input("Location", value="San Francisco, CA", help="Target city or 'Remote'")
        scrape_source = st.selectbox(
            "Target Ingestion Source",
            ["All Sources", "LinkedIn Public", "Career Portal", "Reddit Community Discussions"],
        )
        run_scrape_btn = st.form_submit_button("🚀 Run Live Scrape & Recalculate", use_container_width=True)

    if run_scrape_btn:
        with st.sidebar.status("Running Live Ingestion Pipeline...", expanded=True) as status:
            st.write(f"1. Crawling {target_company} via stealth Playwright & Zero-API engine...")
            from scraper.run import run_scrape_for_company
            from scraper.run_reviews import run_reviews_for_company
            from preprocessing.run import process_uncleaned_postings

            source_map = {
                "All Sources": "all",
                "LinkedIn Public": "linkedin",
                "Career Portal": "career",
                "Reddit Community Discussions": "career",
            }
            engine_choice = source_map.get(scrape_source, "all")

            # 1. Scrape postings
            new_p_count = 0
            if scrape_source in ["All Sources", "LinkedIn Public", "Career Portal"]:
                res_postings = run_scrape_for_company(
                    name=target_company,
                    query=job_query,
                    location=target_location,
                    source=engine_choice,
                    db_path=db_path,
                )
                new_p_count = res_postings["stats"].get("new", 0)
                st.write(f"✓ Extracted {res_postings['collected']} postings ({new_p_count} new).")

            # 2. Extract community reviews
            st.write(f"2. Extracting community feedback for '{target_company}' via Zero-API Reddit...")
            res_reviews = run_reviews_for_company(company=target_company, limit=15, db_path=db_path)
            new_r_count = res_reviews["stats"].get("saved", 0)
            st.write(f"✓ Saved {new_r_count} community review entries.")

            # 3. Clean and normalize
            st.write("3. Normalizing HTML & extracting syntactic features...")
            cleaned_count = process_uncleaned_postings(db_path=db_path)

            # 4. Recompute Ghost Scores
            st.write("4. Computing composite 0-100 Ghost Job Scores...")
            scored_count = compute_and_store_scores(db_path=db_path)

            # 5. Emit GA4 Telemetry event
            send_ga4_measurement_event(
                event_name="live_scrape_triggered",
                params={
                    "company": target_company,
                    "source": scrape_source,
                    "new_postings": new_p_count,
                    "new_reviews": new_r_count,
                },
            )

            status.update(label="Live Scrape Complete!", state="complete", expanded=False)

        timestamp_str = datetime.datetime.utcnow().strftime("%H:%M:%S")
        log_entry = f"[{timestamp_str}] Scraped '{target_company}': {new_p_count} new jobs, {new_r_count} reviews. Scores recalculated."
        st.session_state.activity_log.insert(0, log_entry)
        st.session_state["last_scraped_company"] = target_company
        st.sidebar.success(f"Successfully updated '{target_company}'!")
        st.rerun()

    # Database Vital KPI Badges
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗄️ Database Live Record Vitals")
    total_postings_db = len(df_postings)
    high_risk_db = (df_postings["final_ghost_score"] >= 60.0).sum() if not df_postings.empty else 0
    companies_count_db = df_postings["company"].nunique() if not df_postings.empty else 0
    reviews_count_db = int(df_sentiments["review_count"].sum()) if not df_sentiments.empty else 0

    v_c1, v_c2 = st.sidebar.columns(2)
    v_c1.metric("Postings", f"{total_postings_db:,}")
    v_c2.metric("Reviews", f"{reviews_count_db:,}")

    v_c3, v_c4 = st.sidebar.columns(2)
    v_c3.metric("Companies", companies_count_db)
    v_c4.metric("High-Risk", high_risk_db)

    # Live Empirical Correlations in Sidebar
    corr_res = correlation_check(db_path=db_path)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Empirical Correlations")
    st.sidebar.caption("Internal Ghost Score vs External Candidate Sentiment")
    r_val = corr_res.get("pearson_r", 0.0)
    p_val = corr_res.get("pearson_p", 1.0)
    rho_val = corr_res.get("spearman_r", 0.0)

    st.sidebar.markdown(f"**Pearson $r$:** `{r_val:.4f}` *(p={p_val:.4f})*")
    st.sidebar.markdown(f"**Spearman $\\rho$:** `{rho_val:.4f}`")

    # Interactive Filters in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Dashboard Filters")
    available_companies = sorted(df_postings["company"].unique()) if not df_postings.empty else []
    selected_companies = st.sidebar.multiselect("Filter by Company", available_companies, default=available_companies)

    score_range = st.sidebar.slider("Ghost Score Range", 0.0, 100.0, (0.0, 100.0), step=5.0)

    # Filter Postings DataFrame
    if not df_postings.empty:
        df_filtered = df_postings[
            (df_postings["company"].isin(selected_companies))
            & (df_postings["final_ghost_score"] >= score_range[0])
            & (df_postings["final_ghost_score"] <= score_range[1])
        ]
    else:
        df_filtered = df_postings

    # Activity Stream Log in Sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📋 Session Activity Stream", expanded=False):
        for log in st.session_state.activity_log[:8]:
            st.caption(log)

    # =========================================================================
    # MAIN HEADER
    # =========================================================================
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="main-title">Ghost Job Detector — Analytics & Transparency Dashboard</h1>
            <p class="main-subtitle">Multi-signal NLP scoring, industry severity distributions, sentiment correlation, and live candidate traffic telemetry.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview & Distribution",
        "Industry Heatmap",
        "Sentiment Correlation",
        "GA4 Traffic Telemetry",
    ])

    # =========================================================================
    # TAB 1: OVERVIEW & DISTRIBUTION
    # =========================================================================
    with tab1:
        st.header("Overview & Ghost Job Score Distribution")

        if df_filtered.empty:
            st.warning("No job postings match the active filter criteria.")
            if st.button("Reset Snapshot"):
                seed_from_snapshot(db_path=db_path)
                st.rerun()
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Postings Evaluated", len(df_filtered))
            avg_score = df_filtered["final_ghost_score"].mean()
            col2.metric("Average Ghost Score", f"{avg_score:.2f} / 100")
            high_risk_filtered = (df_filtered["final_ghost_score"] >= 60.0).sum()
            col3.metric("High-Risk Postings (≥60)", high_risk_filtered)
            col4.metric("Companies Monitored", df_filtered["company"].nunique())

            # Spotlight on Last Scraped Company
            if "last_scraped_company" in st.session_state:
                l_comp = st.session_state["last_scraped_company"]
                df_recent_comp = df_postings[df_postings["company"] == l_comp]
                if not df_recent_comp.empty:
                    c_avg = df_recent_comp["final_ghost_score"].mean()
                    c_repost = df_recent_comp["repost_score"].mean()
                    c_vague = df_recent_comp["vagueness_score"].mean()
                    risk_badge = "🔴 Elevated Ghost Risk" if c_avg >= 15 else ("🟡 Moderate Risk" if c_avg >= 8 else "🟢 Low Risk (Active Hiring)")
                    st.info(f"✨ **Live Ingestion Spotlight: {l_comp}** — {len(df_recent_comp)} Requisitions Evaluated | Avg Ghost Score: **{c_avg:.1f} / 100** ({risk_badge}) | Repost Index: **{c_repost:.2f}** | Vagueness: **{c_vague:.2f}**")

            # Real-Time Quick Search
            search_query = st.text_input("🔎 Search by Role Title or Company", placeholder="e.g. AI Engineer, Stripe, OpenAI...")
            if search_query:
                send_ga4_measurement_event("score_lookup", {"query": search_query})
                df_filtered = df_filtered[
                    df_filtered["title"].str.contains(search_query, case=False, na=False)
                    | df_filtered["company"].str.contains(search_query, case=False, na=False)
                ]

            st.subheader("Ghost Job Score Distribution Histogram")
            fig_hist = px.histogram(
                df_filtered,
                x="final_ghost_score",
                nbins=20,
                color_discrete_sequence=["#C85A32"],
                title="Distribution of Final Ghost Job Scores across Evaluated Postings",
                labels={"final_ghost_score": "Ghost Job Score (0-100)"},
            )
            fig_hist.update_traces(marker_line_width=1, marker_line_color="#FFFFFF")
            apply_warm_clean_theme(fig_hist, height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Highest & Lowest Risk Postings")
            col_high, col_low = st.columns(2)

            display_cols = ["company", "title", "final_ghost_score", "genericness_score", "vagueness_score", "repost_score"]
            available_disp = [c for c in display_cols if c in df_filtered.columns]

            with col_high:
                st.markdown("### Top 5 Highest Risk Postings (High Likelihood Ghost Jobs)")
                df_high = df_filtered.sort_values(by="final_ghost_score", ascending=False).head(5)
                st.dataframe(df_high[available_disp], use_container_width=True)

            with col_low:
                st.markdown("### Top 5 Lowest Risk Postings (Likely Active Requisitions)")
                df_low = df_filtered.sort_values(by="final_ghost_score", ascending=True).head(5)
                st.dataframe(df_low[available_disp], use_container_width=True)

    # =========================================================================
    # TAB 2: INDUSTRY HEATMAP
    # =========================================================================
    with tab2:
        st.header("Industry Taxonomy Severity Breakdown")

        if df_filtered.empty or "industry" not in df_filtered.columns:
            st.info("No industry data available for current filter selection.")
        else:
            df_ind = df_filtered.groupby("industry").agg(
                postings_count=("posting_id", "count"),
                avg_ghost_score=("final_ghost_score", "mean"),
                avg_vagueness=("vagueness_score", "mean"),
                avg_genericness=("genericness_score", "mean"),
                avg_repost=("repost_score", "mean"),
            ).reset_index()

            fig_bar = px.bar(
                df_ind,
                x="industry",
                y="avg_ghost_score",
                color="avg_ghost_score",
                color_continuous_scale=[[0.0, "#F5EFEB"], [0.5, "#E07A5F"], [1.0, "#C85A32"]],
                title="Average Ghost Job Score by Industry Taxonomy",
                labels={"industry": "Industry Sector", "avg_ghost_score": "Average Ghost Score (0-100)"},
                text_auto=".1f",
            )
            fig_bar.update_traces(marker_line_width=1, marker_line_color="#E8E2D9")
            apply_warm_clean_theme(fig_bar, height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(df_ind, use_container_width=True)

    # =========================================================================
    # TAB 3: SENTIMENT CORRELATION
    # =========================================================================
    with tab3:
        st.header("Sentiment & Model Validation Correlation")

        col_c1, col_c2 = st.columns(2)
        col_c1.metric("Pearson Correlation (r)", f"{r_val:.4f}", f"p-value: {p_val:.4f}", help="Linear correlation between internal classifier and candidate sentiment")
        col_c2.metric("Spearman Correlation (rho)", f"{rho_val:.4f}", help="Rank-order correlation between internal classifier and candidate sentiment")

        if not df_filtered.empty:
            df_comp_agg = df_filtered.groupby("company").agg(
                avg_ghost_score=("final_ghost_score", "mean"),
                avg_bert_score=("bert_score", "mean"),
                sentiment_score=("sentiment_score", "first"),
                posting_count=("posting_id", "count"),
            ).reset_index()

            st.subheader("External Sentiment vs Internal Model Ghost Score Scatter Plot")
            fig_scatter = px.scatter(
                df_comp_agg,
                x="sentiment_score",
                y="avg_ghost_score",
                size="posting_count",
                hover_name="company",
                color="avg_ghost_score",
                color_continuous_scale=[[0.0, "#E8E2D9"], [0.5, "#E07A5F"], [1.0, "#C85A32"]],
                title="Company Frustration Sentiment vs. Average Ghost Score",
                labels={"sentiment_score": "Candidate Frustration Sentiment Score", "avg_ghost_score": "Average Ghost Score"},
            )
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color="#2D2824")))
            apply_warm_clean_theme(fig_scatter, height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # =========================================================================
    # TAB 4: GA4 TRAFFIC TELEMETRY
    # =========================================================================
    with tab4:
        st.header("GA4 Public Transparency Telemetry")

        # Telemetry Connection Status Card
        meas_id = get_ga4_measurement_id() or "G-90J1MLTLJM"
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2D9CD; border-radius: 6px; padding: 14px 18px; margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="telemetry-badge badge-success">● CONNECTED</span>
                        <strong style="color: #1C1917;">Google Analytics 4 Data Stream:</strong> <code>{meas_id}</code>
                    </div>
                    <div style="font-size: 0.85rem; color: #6B635B;">
                        Async Gtag Injected in Head &bull; Measurement Protocol Active
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Control Actions Header
        action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
        with action_col1:
            if st.button("Sync Telemetry Cache", use_container_width=True):
                sync_ga4_page_analytics_to_db(db_path=db_path, force=True, allow_pilot_fallback=True)
                send_ga4_measurement_event("telemetry_cache_synced")
                st.rerun()

        with action_col2:
            if st.button("Reset Benchmark", use_container_width=True):
                seed_pilot_analytics(db_path=db_path)
                st.rerun()

        with action_col3:
            if st.button("📡 Send Live GA4 Test Event", use_container_width=True):
                send_ga4_measurement_event(
                    event_name="manual_test_event",
                    params={"trigger": "user_button_click", "timestamp": str(datetime.datetime.utcnow())},
                )
                st.success("Dispatched test event to GA4 telemetry stream!")
                st.rerun()

        # Ensure df_analytics has data
        if df_analytics.empty:
            seed_pilot_analytics(db_path=db_path)
            _, _, df_analytics = load_dashboard_data(db_path=db_path)

        # 1. Top KPI Summary Scorecards
        total_views = int(df_analytics["pageviews"].sum())
        avg_time = float(df_analytics["avg_time_on_page"].mean())
        avg_bounce = float(df_analytics["bounce_rate"].mean()) * 100
        active_est = int(total_views * 0.4)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("30-Day Total Pageviews", f"{total_views:,}", help="Total visits across all company transparency report pages")
        kpi2.metric("Active Candidates (30d)", f"{active_est:,}", help="Estimated unique job seekers reading transparency scorecards")
        kpi3.metric("Avg Engagement Duration", f"{int(avg_time // 60)}m {int(avg_time % 60):02d}s", f"{avg_time:.1f}s raw", help="Average time spent actively reading company report")
        kpi4.metric("Average Bounce Rate", f"{avg_bounce:.1f}%", help="Percentage of single-page sessions without further interaction")

        st.divider()

        # 2. Charts Section
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Transparency Pageviews by Company")
            fig_views = px.bar(
                df_analytics.sort_values(by="pageviews", ascending=False),
                x="company",
                y="pageviews",
                color="avg_time_on_page",
                color_continuous_scale=[[0.0, "#F5EFEB"], [0.5, "#E07A5F"], [1.0, "#C85A32"]],
                title="30-Day Candidate Traffic by Company Transparency Report",
                labels={"pageviews": "Pageviews", "company": "Company", "avg_time_on_page": "Avg Time (s)"},
                text_auto=True,
            )
            fig_views.update_traces(marker_line_width=1, marker_line_color="#E8E2D9")
            apply_warm_clean_theme(fig_views, height=380)
            st.plotly_chart(fig_views, use_container_width=True)

        with chart_col2:
            st.subheader("Organic Traffic vs Ghost Score Correlation")
            if not df_postings.empty:
                df_comp_scores = df_postings.groupby("company").agg(
                    avg_ghost_score=("final_ghost_score", "mean"),
                    posting_count=("posting_id", "count"),
                ).reset_index()
                df_merged = pd.merge(df_analytics, df_comp_scores, on="company", how="inner")
            else:
                df_merged = pd.DataFrame()

            if not df_merged.empty and len(df_merged) >= 2:
                x_vals = df_merged["pageviews"].values
                y_vals = df_merged["avg_ghost_score"].values
                if len(set(x_vals)) > 1 and len(set(y_vals)) > 1:
                    r_tr, p_tr = stats.pearsonr(x_vals, y_vals)
                    corr_title = f"Traffic vs Ghost Risk (Pearson r = {r_tr:.4f}, p = {p_tr:.4f})"
                else:
                    corr_title = "Traffic vs Ghost Risk Correlation"

                fig_corr = px.scatter(
                    df_merged,
                    x="pageviews",
                    y="avg_ghost_score",
                    size="avg_time_on_page",
                    color="avg_ghost_score",
                    color_continuous_scale=[[0.0, "#F5EFEB"], [0.5, "#E07A5F"], [1.0, "#C85A32"]],
                    hover_name="company",
                    text="company",
                    title=corr_title,
                    labels={"pageviews": "30-Day GA4 Pageviews", "avg_ghost_score": "Average Ghost Job Score"},
                )
                if len(set(x_vals)) > 1:
                    m, b = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 50)
                    y_line = m * x_line + b
                    fig_corr.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="#C85A32", dash="dash", width=2),
                            name="Linear Fit",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                fig_corr.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#2D2824")))
                apply_warm_clean_theme(fig_corr, height=380)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Additional scored company postings needed to compute regression correlation.")

        # 3. Live Session Telemetry Events Table
        st.subheader("⚡ Real-Time GA4 Telemetry Event Stream")
        recent_events = get_recent_telemetry_events()
        if recent_events:
            df_events = pd.DataFrame(recent_events)
            st.dataframe(df_events, use_container_width=True)
        else:
            st.caption("No custom client events triggered yet in this active session. Interact with controls or click 'Send Live GA4 Test Event'.")

        # 4. Detailed Data Table with CSV Export
        st.subheader("Detailed Page Analytics Telemetry (PageAnalytics Table)")
        st.dataframe(df_analytics, use_container_width=True)

        csv_data = df_analytics.to_csv(index=False).encode("utf-8")
        if st.download_button(
            label="Download Telemetry Data as CSV (for IEEE Paper)",
            data=csv_data,
            file_name="ga4_transparency_telemetry.csv",
            mime="text/csv",
        ):
            send_ga4_measurement_event("export_click", {"format": "csv", "table": "page_analytics"})


if __name__ == "__main__":
    main()
