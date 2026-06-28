import streamlit as st
from riwaaya_lib.data_loader import load_data
from riwaaya_lib.styling import inject_css, ACCENT

st.set_page_config(
    page_title="Riwaaya Analytics Dashboard",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

df = load_data()

st.title("Riwaaya: Consumer Analytics Dashboard")
st.markdown(
    "##### Authenticated Indian handicrafts marketplace: data analytics, "
    "diagnostic insight, and machine learning for go-to-market decisions"
)
st.divider()

# -----------------------------------------------------------------
# TOP-LINE METRICS
# -----------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Survey Respondents", f"{len(df):,}")
col2.metric("Strong Membership Intent", f"{df['target_membership_interest'].mean()*100:.1f}%")
col3.metric("High Intent Customers", f"{df['high_intent_customer'].mean()*100:.1f}%")
col4.metric("Premium Buyers", f"{df['premium_buyer'].mean()*100:.1f}%")
col5.metric("Avg Annual Spend Potential", f"${df['annual_spending_potential_usd'].mean():,.0f}")

st.divider()

st.markdown("### How to use this dashboard")
left, right = st.columns([2, 1])
with left:
    st.markdown(
        """
        Use the sidebar to move through the analysis. Each page builds on the last:

        1. **Descriptive Analytics**: who the respondents are, how they buy, what they'll pay,
           what they want, and what they trust.
        2. **Diagnostic Analytics**: what actually drives membership interest and spending,
           broken down by age, income, customer type, heritage connection, and geography,
           with significance testing so the patterns aren't just noise.
        3. **Predictive Modeling**: four classification algorithms (KNN, Decision Tree,
           Random Forest, Gradient Boosting), tuned and cross-validated, predicting who
           will commit to a Riwaaya membership.
        4. **Predictive Analytics**: regression models forecasting annual spending
           potential and future purchase frequency, plus a conversion-likelihood scoring
           tool.
        5. **Prescriptive Strategy**: what Riwaaya should actually do about all of this:
           segmentation, pricing, marketing, membership design, and expansion priorities,
           closing with a sustainability verdict.
        """
    )
with right:
    st.markdown(
        f"""
        <div class="insight-box">
        <b>About this data</b><br><br>
        Synthetic consumer survey built to validate the Riwaaya concept: a premium,
        authenticated handicrafts marketplace. 1,000 respondents across five hypothesized
        segments, covering demographics, buying behavior, willingness to pay, trust
        requirements, and stated purchase intent.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Riwaaya Analytics Dashboard · Individual & Group PBL, Data Analytics for Insights and Decision Making")
