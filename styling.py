"""
Shared visual style for the Riwaaya dashboard.
One accent color, two neutrals, kept minimal and consistent across every page.
"""
import plotly.io as pio
import plotly.graph_objects as go

ACCENT = "#A6432F"       # terracotta, the Riwaaya brand accent
ACCENT_LIGHT = "#D98C7A"
DARK = "#2E2E2E"
GRAY = "#B8B8B8"
GRAY_LIGHT = "#EDEAE6"
BG = "#FFFFFF"

PALETTE = [ACCENT, DARK, GRAY, ACCENT_LIGHT, "#6B6B6B", "#E0C9A6"]

CUSTOM_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Helvetica, Arial, sans-serif", color=DARK, size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        colorway=PALETTE,
        title=dict(font=dict(size=18, color=DARK), x=0.02, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRAY, ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor=GRAY_LIGHT, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
)
pio.templates["riwaaya"] = CUSTOM_TEMPLATE
pio.templates.default = "riwaaya"


def inject_css():
    import streamlit as st
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        h1, h2, h3 {{ color: {DARK}; font-family: Helvetica, Arial, sans-serif; }}
        [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
        [data-testid="stSidebar"] {{ background-color: {GRAY_LIGHT}; }}
        .insight-box {{
            background-color: {GRAY_LIGHT};
            border-left: 4px solid {ACCENT};
            padding: 14px 18px;
            border-radius: 4px;
            margin: 10px 0px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
