"""Phase 7 dashboard, screen 4: Fairness-over-time across the synthetic
periods, plus Phase 4's fairness-accuracy trade-off table (roadmap 7.2 #4).
Pure presentation over precomputed Phase 4 CSVs (src/app/state.py,
src/app/routers/fairness.py) -- no new modeling.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import get_fairness_by_period, get_fairness_isolated_effect, get_fairness_tradeoff

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
PERIOD_ORDER = ["P0", "P1", "P2", "P3", "P4"]

st.set_page_config(page_title="Fairness Over Time", page_icon="⚖️", layout="wide")
st.title("Fairness Over Time")

try:
    cross_period = get_fairness_by_period()
    isolated = get_fairness_isolated_effect()
except Exception as e:
    st.error(f"Could not reach the backend API. Is `uvicorn main:app` running? ({e})")
    st.stop()

cross_period = sorted(cross_period, key=lambda r: PERIOD_ORDER.index(r["period"]))
isolated = sorted(isolated, key=lambda r: PERIOD_ORDER.index(r["period"]))

st.subheader("Naive cross-period comparison (context only)")
st.warning(
    "This compares fairness metrics across P0 (clean) vs. P1–P4 (biased) directly. "
    "Each period's independent covariate-drift resampling reshuffles who's in the "
    "population, which moves these numbers around for reasons that have nothing to do "
    "with the planted gender bias -- so this view looks noisy/non-monotonic even though "
    "the bias is real and detectable. See the isolated-effect section below for the "
    "actual finding."
)

col_a, col_b = st.columns([1.6, 1])

with col_a:
    periods = [r["period"] for r in cross_period]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=[r["DPD"] for r in cross_period], mode="lines+markers", name="DPD", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=periods, y=[r["EOD"] for r in cross_period], mode="lines+markers", name="EOD", line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=periods, y=[r["EqualizedOddsDiff"] for r in cross_period], mode="lines+markers", name="Equalized Odds Diff", line=dict(color=AQUA, width=2)))
    fig.update_layout(
        yaxis_title="Metric value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    fig_flip = go.Figure()
    fig_flip.add_trace(go.Bar(x=periods, y=[r["flip_rate"] for r in cross_period], marker_color=MUTED, width=0.5))
    fig_flip.update_layout(
        title="Injected label-flip rate (F, per period)",
        yaxis_tickformat=".0%",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=40),
        showlegend=False,
    )
    st.plotly_chart(fig_flip, width="stretch")

st.divider()

st.subheader("Isolated bias effect (the real finding)")
st.caption(
    "Same population, same model decisions, per biased period -- fairness metrics computed "
    "against TARGET_original (pre-flip) vs TARGET (post-flip). This isolates exactly what the "
    "planted bias contributes, holding covariate drift fixed."
)

all_dpd_equal = all(r["DPD_true"] == r["DPD_biased"] for r in isolated)
st.info(
    f"DPD_true == DPD_biased in every period: **{all_dpd_equal}** -- expected, since DPD never "
    "reads TARGET and is structurally blind to a label-only bias."
)

fig2 = go.Figure()
fig2.add_trace(
    go.Bar(
        x=[r["period"] for r in isolated],
        y=[r["F_FPR_inflation"] for r in isolated],
        marker_color=BLUE,
        text=[f"+{r['F_FPR_inflation']:.1%}" for r in isolated],
        textposition="outside",
        width=0.5,
    )
)
fig2.update_layout(
    yaxis_title="F false-positive-rate inflation",
    yaxis_tickformat=".0%",
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    margin=dict(t=10),
    showlegend=False,
)
st.plotly_chart(fig2, width="stretch")
st.caption(
    "F_FPR_inflation = F's false-positive rate under the biased TARGET minus under "
    "TARGET_original. Consistently positive across all 4 biased periods -- outcome-aware "
    "fairness metrics do catch this failure mode when a clean reference is available."
)

st.divider()

st.subheader("Fairness-accuracy trade-off")
attribute = st.selectbox(
    "Attribute",
    options=["age_band", "CODE_GENDER"],
    help="age_band has the largest real disparity and is the illustrative case; "
    "CODE_GENDER's real disparity was too small to be illustrative (kept for completeness).",
)

try:
    tradeoff = get_fairness_tradeoff(attribute)
except Exception as e:
    st.error(f"Could not reach the backend API. ({e})")
    st.stop()

STRATEGY_LABELS = {
    "baseline": "Baseline (single global threshold)",
    "equalize_dp": "Targeted: equalize DP",
    "equalize_eo": "Targeted: equalize EO",
    "blunt_stricter": "Naive: blunt stricter threshold",
}
STRATEGY_COLORS = {"baseline": MUTED, "equalize_dp": BLUE, "equalize_eo": AQUA, "blunt_stricter": ORANGE}
baseline_profit = next(r["profit"] for r in tradeoff if r["strategy"] == "baseline")

col_c, col_d = st.columns([1.3, 1])

with col_c:
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=[STRATEGY_LABELS[r["strategy"]] for r in tradeoff],
            y=[r["DPD"] for r in tradeoff],
            marker_color=[STRATEGY_COLORS[r["strategy"]] for r in tradeoff],
            text=[f"{r['DPD']:.3f}" for r in tradeoff],
            textposition="outside",
        )
    )
    fig3.update_layout(
        yaxis_title="DPD",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=10),
        showlegend=False,
    )
    st.plotly_chart(fig3, width="stretch")

with col_d:
    st.dataframe(
        [
            {
                "Strategy": STRATEGY_LABELS[r["strategy"]],
                "DPD": round(r["DPD"], 4),
                "EOD": round(r["EOD"], 4),
                "Profit vs. baseline": f"{(r['profit'] - baseline_profit) / baseline_profit:+.2%}",
            }
            for r in tradeoff
        ],
        hide_index=True,
        width="stretch",
    )

st.caption(
    "Targeted group-specific threshold mitigation closes most of the disparity for a "
    "negligible profit cost; the naive blunt stricter threshold costs far more profit and "
    "can make DPD/EOD *worse*, not better -- a concrete 'blunt mitigation can backfire' finding."
)
