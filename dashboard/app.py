import os
import sys
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from scipy import stats

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scraper.models import init_db, get_session, JobPosting, GhostScore, CompanySentiment, CompanyReview, PageAnalytics
from scoring.final_score import correlation_check, DEFAULT_SCORE_WEIGHTS
from analytics.ga4_client import (
    get_ga4_traffic_metrics,
    sync_ga4_page_analytics_to_db,
    is_ga4_configured,
    seed_pilot_analytics,
    fetch_ga4_30day_page_metrics,
)

# Streamlit Page Configuration (no emojis, light layout)
st.set_page_config(
    page_title="Ghost Job Detector — Analytics & Transparency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        background: linear-gradient(180deg, #FFFFFF 0%, #ECE6DC 100%) !important;
        color: #1C1917 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 5px rgba(60, 50, 40, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #FFFDFB 0%, #FAF5EE 100%) !important;
        color: #C85A32 !important;
        border: 1px solid #C85A32 !important;
        border-top: 3px solid #C85A32 !important;
        border-bottom: 1px solid #FAF5EE !important;
        box-shadow: 0 -1px 3px rgba(200, 90, 50, 0.08), 0 2px 0 #FAF5EE !important;
        transform: translateY(1px) !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Metric Cards: Crisp Warm Surface */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E8E2D9 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        box-shadow: 0 1px 3px rgba(60, 50, 40, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 0.82rem !important;
        color: #6B635B !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 1.45rem !important;
        color: #1C1917 !important;
        font-weight: 700 !important;
    }

    /* Buttons: Tactile Warm Button */
    div.stButton > button {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# Live GA4 telemetry tracking tag
ga4_meas_id = os.getenv("GA4_MEASUREMENT_ID")
if ga4_meas_id and "your_" not in ga4_meas_id.lower():
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


def load_dashboard_data(db_path: str = "data/ghostjobs.db"):
    """Load live data from SQLite database into pandas DataFrames."""
    if not os.path.exists(db_path):
        init_db(db_path)

    session = get_session(db_path)
    try:
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
    # Compact Header
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="main-title">Ghost Job Detector — Analytics & Transparency Dashboard</h1>
            <p class="main-subtitle">Multi-signal NLP scoring, industry severity distributions, sentiment correlation, and candidate traffic telemetry.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db_path = "data/ghostjobs.db"
    sync_ga4_page_analytics_to_db(db_path=db_path, allow_pilot_fallback=True)
    df_postings, df_sentiments, df_analytics = load_dashboard_data(db_path=db_path)
    industry_mapping = load_company_industries()

    if not df_postings.empty:
        df_postings["industry"] = df_postings["company"].map(lambda c: industry_mapping.get(c, "General Tech"))

    # Navigation Tabs (No emojis, 3D tactile button styling)
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

        if df_postings.empty:
            st.warning("No scored job postings found in the database. Run `python scoring/run.py` to populate scores.")
        else:
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
                color_discrete_sequence=["#C85A32"],
                title="Distribution of Final Ghost Job Scores across Analyzed Postings",
                labels={"final_ghost_score": "Ghost Job Score (0-100)"},
            )
            fig_hist.update_traces(marker_line_width=1, marker_line_color="#FFFFFF")
            apply_warm_clean_theme(fig_hist, height=360)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Highest & Lowest Scoring Postings")
            col_high, col_low = st.columns(2)

            with col_high:
                st.markdown("### Top 5 Highest Risk Postings (High Likelihood Ghost Jobs)")
                df_high = df_postings.sort_values(by="final_ghost_score", ascending=False).head(5)
                st.dataframe(
                    df_high[["company", "title", "final_ghost_score", "genericness_score", "vagueness_score", "repost_score"]],
                    use_container_width=True,
                )

            with col_low:
                st.markdown("### Top 5 Lowest Risk Postings (Likely Legitimate)")
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
                color_continuous_scale=[[0.0, "#F5EFEB"], [0.5, "#E07A5F"], [1.0, "#C85A32"]],
                title="Average Ghost Job Score Severity by Industry",
                labels={"avg_ghost_score": "Average Ghost Score", "industry": "Industry Sector"},
                text_auto=".2f",
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

        corr_res = correlation_check(db_path=db_path)

        col_c1, col_c2 = st.columns(2)
        col_c1.metric("Pearson Correlation (r)", f"{corr_res['pearson_r']:.4f}", help="Linear correlation between internal classifier and candidate sentiment")
        col_c2.metric("Spearman Correlation (rho)", f"{corr_res['spearman_r']:.4f}", help="Rank-order correlation between internal classifier and candidate sentiment")

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
        st.markdown(
            "Empirical validation layer tracking real-world candidate search traffic to public hiring transparency scorecards. "
            "Correlates external visitor interest with internal composite ghost job risk scores (IEEE Study Stage 7)."
        )

        is_live = is_ga4_configured()

        # Status & Control Header
        status_col, action_col1, action_col2 = st.columns([2, 1, 1])
        with status_col:
            if is_live:
                st.success("Live GA4 Connected: Pulling live visitor telemetry from Google Analytics 4 Data Stream.")
            else:
                st.info("Pilot Benchmark Mode Active: Displaying structured telemetry for research paper evaluation.")

        with action_col1:
            if st.button("Sync Live Data", use_container_width=True):
                sync_ga4_page_analytics_to_db(db_path=db_path, force=True, allow_pilot_fallback=True)
                st.rerun()

        with action_col2:
            if st.button("Reset Benchmark", use_container_width=True):
                seed_pilot_analytics(db_path=db_path)
                st.rerun()

        # Ensure df_analytics has data (fall back to pilot if empty)
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
                    r_val, p_val = stats.pearsonr(x_vals, y_vals)
                    r_val, p_val = float(r_val), float(p_val)
                    corr_title = f"Traffic vs Ghost Risk (Pearson r = {r_val:.4f}, p = {p_val:.4f})"
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
                st.caption("Research Note: Evaluates whether candidates organically search for reports on companies with higher ghost job risk.")
            else:
                st.info("Additional scored company postings needed to compute regression correlation.")

        # 3. Retention & Engagement Comparison
        st.subheader("Candidate Attention and Retention Matrix")
        fig_scatter_att = px.scatter(
            df_analytics,
            x="avg_time_on_page",
            y="bounce_rate",
            size="pageviews",
            color="company",
            color_discrete_sequence=["#C85A32", "#D97706", "#8D5B4C", "#4A5568", "#2B2D42"],
            hover_name="company",
            title="Engagement Duration vs Bounce Rate per Company Report",
            labels={"avg_time_on_page": "Average Time on Page (seconds)", "bounce_rate": "Bounce Rate (ratio)"},
        )
        fig_scatter_att.update_traces(marker=dict(line=dict(width=1, color="#2D2824")))
        apply_warm_clean_theme(fig_scatter_att, height=360)
        st.plotly_chart(fig_scatter_att, use_container_width=True)

        # 4. Detailed Data Table with CSV Export
        st.subheader("Detailed Page Analytics Telemetry (PageAnalytics)")
        st.dataframe(df_analytics, use_container_width=True)

        csv_data = df_analytics.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Telemetry Data as CSV (for IEEE Paper)",
            data=csv_data,
            file_name="ga4_transparency_telemetry.csv",
            mime="text/csv",
        )

        st.divider()

        # 5. Live GA4 Connection & Setup Guide (Expander)
        with st.expander("Connect Live Google Analytics 4 (Setup Guide and Configuration)", expanded=False):
            st.markdown("""
            ### How to Connect Real Google Analytics 4 (GA4)
            Follow these steps to link your live tracking:

            #### Step 1: Create a GA4 Property
            1. Go to Google Analytics and create a GA4 Property.
            2. Add a Web Data Stream and copy your Measurement ID (format: `G-XXXXXXXXXX`).

            #### Step 2: Configure Environment Variables
            Open `.env` in the project root and update:
            ```env
            GA4_MEASUREMENT_ID=G-XXXXXXXXXX
            GA4_PROPERTY_ID=123456789
            ```
            *(Optional for direct API sync)* Set `GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json`.

            #### Step 3: Bake Tracking into Static SEO Pages
            Run the static transparency generator:
            ```bash
            python seo/build.py --threshold 60
            ```
            This automatically injects the `gtag.js` tracking snippet into every generated HTML page in `seo/build/`.

            #### Step 4: Deploy Static Site to GitHub Pages
            1. Push changes to GitHub.
            2. Go to Repo Settings -> Pages -> Source: Deploy from branch (`main` / `seo/build`).
            3. As candidates search and visit `https://<username>.github.io/ghost-job-detector/`, GA4 will automatically record telemetry.
            """)

            st.markdown("#### Test In-Memory Connection")
            test_col1, test_col2 = st.columns(2)
            with test_col1:
                input_meas_id = st.text_input("GA4 Measurement ID", value=os.getenv("GA4_MEASUREMENT_ID", ""))
            with test_col2:
                input_prop_id = st.text_input("GA4 Property ID", value=os.getenv("GA4_PROPERTY_ID", ""))

            if st.button("Test and Save Live GA4 Connection"):
                if input_meas_id and "your_" not in input_meas_id.lower():
                    os.environ["GA4_MEASUREMENT_ID"] = input_meas_id
                    if input_prop_id:
                        os.environ["GA4_PROPERTY_ID"] = input_prop_id
                    new_count = sync_ga4_page_analytics_to_db(db_path=db_path, force=True, allow_pilot_fallback=False)
                    st.success(f"Connection test complete! Synced {new_count} metrics.")
                    st.rerun()
                else:
                    st.error("Please enter a valid GA4 Measurement ID (format: G-XXXXXXXXXX).")


if __name__ == "__main__":
    main()
