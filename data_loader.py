"""
Central data loading and feature engineering for the Riwaaya dashboard.
Every page imports load_data() so the cleaning logic only lives in one place.
"""
import os
import re
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "riwaaya_survey_cleaned.csv")

REGION_MAP = {
    "UAE": "GCC",
    "India": "South Asia",
    "UK": "Europe", "Germany": "Europe", "Switzerland": "Europe", "Italy": "Europe",
    "France": "Europe", "Netherlands": "Europe", "Sweden": "Europe",
    "USA": "North America", "Canada": "North America",
    "Singapore": "APAC", "Australia": "APAC", "Hong Kong": "APAC",
}

INCOME_ORDER = ["Under $25K", "$25-50K", "$50-100K", "$100-200K", "$200K+", "Prefer not to say"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]
ROLE_ORDER_BY_INTEREST = None  # computed at runtime, kept for reference


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the cleaned survey CSV and add the few derived fields every
    page needs: a readable region grouping and the binary membership target."""
    df = pd.read_csv(DATA_PATH)
    df["region"] = df["country"].map(REGION_MAP)

    # Target: "Yes, definitely" vs everything else. Close to a 50/50 split
    # (50.5% / 49.5%), which is what makes it a clean classification target
    # rather than a heavily imbalanced one.
    df["target_membership_interest"] = (df["membership_interest"] == "Yes, definitely").astype(int)
    df["wants_launch_notification_int"] = df["wants_launch_notification"].astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, drop_cols=None):
    """Builds the numeric feature matrix used by every model on the
    Predictive Modeling and Predictive Analytics pages. drop_cols lets a
    specific model exclude features that would otherwise leak the target
    (e.g. excluding future_purchases_per_year when that IS the target)."""
    drop_cols = set(drop_cols or [])

    numeric_features = [
        "age_range_ordinal", "income_bracket_ordinal", "items_purchased_12m_ordinal",
        "items_purchased_12m_raw", "appeal_score", "trust_count", "nps_score",
        "max_price_small_piece_usd", "max_price_large_piece_usd", "authenticity_premium_pct",
        "future_purchases_per_year", "annual_spending_potential_usd",
        "high_intent_customer", "premium_buyer", "high_trust_customer",
        "wants_launch_notification_int",
    ]
    numeric_features = [c for c in numeric_features if c not in drop_cols]

    raw_or_numeric_trust_cols = {"buying_channels", "product_interests", "trust_factors", "trust_count"}
    existing_onehot = [
        c for c in df.columns
        if c.startswith(("channel_", "interest_", "trust_")) and c not in raw_or_numeric_trust_cols
    ]

    nominal_to_encode = ["gender", "region", "respondent_role", "heritage_connection",
                          "top_frustration", "main_purchase_blocker", "top_wishlist_feature"]
    df_encoded = pd.get_dummies(df[nominal_to_encode], prefix=nominal_to_encode, drop_first=False)

    X = pd.concat([df[numeric_features], df[existing_onehot], df_encoded], axis=1)
    return X


def split_multiselect(value):
    if pd.isna(value):
        return []
    return [v.strip() for v in str(value).split(";") if v.strip()]


def safe_name(text):
    text = text.replace("'", "")
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
