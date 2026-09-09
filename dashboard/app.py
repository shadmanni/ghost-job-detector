import os
import sys
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanySentiment, CompanyReview
from scoring.final_score import correlation_check, DEFAULT_SCORE_WEIGHTS
from analytics.ga4_client import get_ga4_traffic_metrics

# Streamlit Page Configuration
st.set_page_config(
    page_title="Ghost Job Detector Dashboard",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def load_dashboard_data(db_path: str = "data/ghostjobs.db"):
    """Load live data from SQLite database into pandas DataFrames."""
    if not os.path.exists(db_path):
        init_db(db_path)

    session = get_session(db_path)
    try:
        # Query JobPostings and GhostScores
        query = (
            session.query(JobPosting, GhostScore)
            .join(GhostScore, JobPosting.id == GhostScore.job_posting_id)
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
                "posted_date": posting.posted_date,
                "scraped_at": posting.scraped_at,
                "genericness_score": score.genericness_score or 0.0,
                "vagueness_score": score.vagueness_score or 0.0,
                "repost_score": score.repost_score or 0.0,
                "urgency_score": score.urgency_score or 0.0,
                "bert_score": score.bert_score or 0.0,
                "sentiment_score": score.sentiment_score or 0.0,
                "final_ghost_score": score.final_score or 0.0,
            })

        df_postings = pd.DataFrame(rows)

        # Query CompanySentiments
        sentiments = session.query(CompanySentiment).all()
        s_rows = [{"company": s.company, "sentiment_score": s.score, "review_count": s.review_count} for s in sentiments]
        df_sentiments = pd.DataFrame(s_rows)

        return df_postings, df_sentiments
    finally:
        session.close()


def main():
    st.title("👻 Ghost Job Detector — Analytics & Transparency Dashboard")
    st.markdown(
        "Real-time analytics, multi-factor ghost signal scoring, industry heatmaps, "
        "and candidate sentiment correlation."
    )

    db_path = "data/ghostjobs.db"
    df_postings, df_sentiments = load_dashboard_data(db_path=db_path)
    industry_mapping = load_company_industries()

    if not df_postings.empty:
        df_postings["industry"] = df_postings["company"].map(lambda c: industry_mapping.get(c, "General Tech"))

    # Create Dashboard Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Distribution",
        "🔥 Industry Heatmap",
        "📈 Sentiment Correlation",
        "🌐 GA4 Traffic Telemetry",
    ])

    # =========================================================================
    # TAB 1: OVERVIEW & DISTRIBUTION
    # =========================================================================
    with tab1:
        st.header("Overview & Ghost Job Score Distribution")

        if df_postings.empty:
            st.warning("No scored job postings found in the database. Run `python scoring/run.py` to populate scores.")
        else:
            # Top KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Postings Analyzed", len(df_postings))
            col2.metric("Average Ghost Score", f"{df_postings['final_ghost_score'].mean():.2f} / 100")
            high_risk_count = (df_postings["final_ghost_score"] >= 60.0).sum()
            col3.metric("High-Risk Ghost Jobs", high_risk_count)
            col4.metric("Companies Tracked", df_postings["company"].nunique())

            st.subheader("Ghost Job Score Distribution Histogram")
            fig_hist = px.histogram(
                df_postings,
                x="final_ghost_score",
                nbins=20,
                color_discrete_sequence=["#38bdf8"],
                title="Distribution of Final Ghost Job Scores across Analyzed Postings",
                labels={"final_ghost_score": "Ghost Job Score (0-100)"},
            )
            fig_hist.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Highest & Lowest Scoring Postings")
            col_high, col_low = st.columns(2)

            with col_high:
                st.markdown("### ⚠️ Top 5 Highest Risk Postings (Likely Ghost Jobs)")
                df_high = df_postings.sort_values(by="final_ghost_score", ascending=False).head(5)
                st.dataframe(
                    df_high[["company", "title", "final_ghost_score", "genericness_score", "vagueness_score", "repost_score"]],
                    use_container_width=True,
                )

            with col_low:
                st.markdown("### ✅ Top 5 Lowest Risk Postings (Likely Legitimate)")
                df_low = df_postings.sort_values(by="final_ghost_score", ascending=True).head(5)
                st.dataframe(
                    df_low[["company", "title", "final_ghost_score", "genericness_score", "vagueness_score", "repost_score"]],
                    use_container_width=True,
                )

    # =========================================================================
    # TAB 2: INDUSTRY HEATMAP
    # =========================================================================
    with tab2:
        st.header("Industry Severity Heatmap")

        if df_postings.empty:
            st.warning("No posting data available for industry heatmap analysis.")
        else:
            df_ind = df_postings.groupby("industry").agg(
                avg_ghost_score=("final_ghost_score", "mean"),
                posting_count=("posting_id", "count"),
                avg_repost_score=("repost_score", "mean"),
                avg_genericness=("genericness_score", "mean"),
            ).reset_index()

            st.markdown("Average Ghost Job Score severity breakdown categorized by company industry taxonomy:")

            fig_bar = px.bar(
                df_ind,
                x="industry",
                y="avg_ghost_score",
                color="avg_ghost_score",
                color_continuous_scale="Reds",
                title="Average Ghost Job Score Severity by Industry",
                labels={"avg_ghost_score": "Average Ghost Score", "industry": "Industry Sector"},
                text_auto=".2f",
            )
            fig_bar.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(df_ind, use_container_width=True)

    # =========================================================================
    # TAB 3: SENTIMENT CORRELATION
    # =========================================================================
    with tab3:
        st.header("Sentiment & Model Validation Correlation")

        corr_res = correlation_check(db_path=db_path)

        col_c1, col_c2 = st.columns(2)
        col_c1.metric("Pearson Correlation (r)", f"{corr_res['pearson_r']:.4f}", help="Linear correlation between internal classifier and candidate sentiment")
        col_c2.metric("Spearman Correlation (ρ)", f"{corr_res['spearman_r']:.4f}", help="Rank-order correlation between internal classifier and candidate sentiment")

        if not df_postings.empty:
            df_comp_agg = df_postings.groupby("company").agg(
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
                color_continuous_scale="Viridis",
                title="Company Frustration Sentiment vs. Average Ghost Score",
                labels={"sentiment_score": "Candidate Frustration Sentiment Score", "avg_ghost_score": "Average Ghost Score"},
            )
            fig_scatter.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # =========================================================================
    # TAB 4: GA4 TRAFFIC TELEMETRY
    # =========================================================================
    with tab4:
        st.header("GA4 Traffic & Visitor Telemetry")

        traffic_data = get_ga4_traffic_metrics()

        if traffic_data is None:
            st.info(
                "ℹ️ **No GA4 Telemetry Data Connected**\n\n"
                "Google Analytics 4 credentials (`GA4_MEASUREMENT_ID` and `GA4_API_SECRET`) are unconfigured or set to placeholders in `.env`.\n"
                "Configure your GA4 API secrets in `.env` to enable real-time website traffic metrics."
            )
        else:
            st.success("Connected to GA4 Data Stream")
            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("30-Day Active Users", traffic_data.get("active_users_30d", 0))
            t_col2.metric("30-Day Total Sessions", traffic_data.get("sessions_30d", 0))
            t_col3.metric("30-Day Page Views", traffic_data.get("page_views_30d", 0))

            st.subheader("Top Transparency Report Page Views")
            df_ga4 = pd.DataFrame(traffic_data.get("top_reports", []))
            if not df_ga4.empty:
                fig_ga4 = px.bar(
                    df_ga4,
                    x="page",
                    y="views",
                    color="views",
                    title="Most Viewed Transparency Reports",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_ga4, use_container_width=True)


if __name__ == "__main__":
    main()
