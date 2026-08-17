"""Phase 7 dashboard, screen 1: Portfolio Overview (roadmap 7.2 #1).
Aggregate stats only -- no per-applicant drill-down here, that's screen 3.
All numbers come from the FastAPI backend (src/app/main.py), never
recomputed in the dashboard process itself.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import get_applicant_page, get_deployed_threshold, get_profit_curve, get_simulation

BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"

st.set_page_config(page_title="Portfolio Overview", page_icon="\U0001F4CA", layout="wide")
st.title("Portfolio Overview")

try:
    total_applicants = get_applicant_page(limit=1)["total"]
    threshold = get_deployed_threshold()
    sim = get_simulation(threshold)
    curve = get_profit_curve()
except Exception as e:
    st.error(f"Could not reach the backend API at the configured API_BASE_URL. Is `uvicorn main:app` running? ({e})")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total applicants", f"{total_applicants:,}")
col2.metric("Decision threshold (PD)", f"{threshold:.3f}")
col3.metric("Approval rate", f"{sim['approval_rate']:.1%}")
col4.metric("Default rate among approved", f"{sim['approved_default_rate']:.1%}")

st.metric("Portfolio profit at current threshold", f"${sim['profit']:,.0f}")

st.divider()

thresholds = [row["threshold"] for row in curve]
profits = [row["profit"] for row in curve]
approval_rates = [row["approval_rate"] for row in curve]
default_rates = [row["approved_default_rate"] for row in curve]

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Profit vs. decision threshold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=profits, mode="lines", line=dict(color=BLUE, width=2), name="Profit"))
    fig.add_vline(x=threshold, line=dict(color=MUTED, width=2, dash="dash"))
    fig.add_annotation(x=threshold, y=max(profits), text="deployed threshold", showarrow=False, yshift=12, font=dict(color=MUTED, size=11))
    fig.update_layout(
        xaxis_title="PD threshold (approve if PD ≤ threshold)",
        yaxis_title="Profit ($)",
        showlegend=False,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=10),
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("Approval rate & default rate vs. threshold")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=thresholds, y=approval_rates, mode="lines", line=dict(color=BLUE, width=2), name="Approval rate"))
    fig2.add_trace(go.Scatter(x=thresholds, y=default_rates, mode="lines", line=dict(color=ORANGE, width=2), name="Default rate (approved)"))
    fig2.add_vline(x=threshold, line=dict(color=MUTED, width=2, dash="dash"))
    fig2.update_layout(
        xaxis_title="PD threshold",
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=10),
    )
    st.plotly_chart(fig2, width="stretch")

st.caption(
    "Profit/approval/default curves are Phase 3's precomputed profit curve for the deployed "
    "LightGBM model; the deployed threshold marker matches the model artifact's stored optimal "
    "threshold, read live from the API rather than duplicated here."
)
