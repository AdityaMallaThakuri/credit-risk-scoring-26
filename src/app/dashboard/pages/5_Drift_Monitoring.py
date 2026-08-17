"""Phase 7 dashboard, screen 5: Drift Monitoring -- PSI/KS/JS by feature,
by period (roadmap 7.2 #5). Pure presentation over Phase 5's precomputed
drift metrics (src/drift/drift_metrics.py, served as-is by
src/app/routers/drift.py) -- no new computation.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import get_drift

FEATURE_COLORS = {
    "EXT_SOURCE_1": "#2a78d6",
    "EXT_SOURCE_2": "#eb6834",
    "EXT_SOURCE_3": "#1baf7a",
    "AMT_INCOME_TOTAL": "#eda100",
}
FEATURE_ORDER = list(FEATURE_COLORS)
PERIOD_ORDER = ["P1", "P2", "P3", "P4"]
GRIDLINE = "#e1e0d9"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

st.set_page_config(page_title="Drift Monitoring", page_icon="\U0001F30A", layout="wide")
st.title("Drift Monitoring")
st.caption(
    "P0 (the reference period) isn't shown -- every metric here measures a period's "
    "drift *against* P0, so P0 vs. itself is trivially zero."
)

try:
    drift = get_drift()
except Exception as e:
    st.error(f"Could not reach the backend API. Is `uvicorn main:app` running? ({e})")
    st.stop()

by_feature = {f: sorted([r for r in drift if r["feature"] == f], key=lambda r: PERIOD_ORDER.index(r["period"])) for f in FEATURE_ORDER}


def _line_chart(metric_key, y_title, ref_lines=None):
    fig = go.Figure()
    for feature in FEATURE_ORDER:
        rows = by_feature[feature]
        fig.add_trace(
            go.Scatter(
                x=[r["period"] for r in rows],
                y=[r[metric_key] for r in rows],
                mode="lines+markers",
                name=feature,
                line=dict(color=FEATURE_COLORS[feature], width=2),
                marker=dict(size=8),
            )
        )
    for y, label, color in ref_lines or []:
        fig.add_hline(y=y, line=dict(color=color, width=1.5, dash="dash"))
        fig.add_annotation(x=PERIOD_ORDER[0], y=y, text=label, showarrow=False, yshift=10, xshift=30, font=dict(color=color, size=11))
    fig.update_layout(
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=40),
    )
    return fig


tab_psi, tab_ks, tab_js = st.tabs(["PSI", "KS statistic", "JS divergence"])

with tab_psi:
    st.plotly_chart(
        _line_chart("psi", "Population Stability Index", ref_lines=[(0.10, "moderate shift", WARNING), (0.25, "major shift", CRITICAL)]),
        width="stretch",
    )
    st.caption(
        "PSI rises monotonically P1→P4 for every feature, matching the synthetic overlay's "
        "logged design targets (0.15 / 0.30 / 0.45 / 0.60) to within measurement tolerance."
    )

with tab_ks:
    st.plotly_chart(_line_chart("ks_statistic", "KS statistic"), width="stretch")
    st.caption(
        "KS p-values are omitted from this chart -- every p-value in this dataset rounds to "
        "0.0 (see the table below), since statistical significance is close to guaranteed at "
        "~75,000 applicants/period regardless of practical drift magnitude. The KS *statistic* "
        "itself is the informative number here."
    )

with tab_js:
    st.plotly_chart(_line_chart("js_divergence", "Jensen-Shannon divergence"), width="stretch")

st.divider()
st.subheader("Full metric table")
st.dataframe(
    [
        {
            "Period": r["period"],
            "Feature": r["feature"],
            "PSI": round(r["psi"], 4),
            "KS statistic": round(r["ks_statistic"], 4),
            "KS p-value": r["ks_pvalue"],
            "JS divergence": round(r["js_divergence"], 4),
        }
        for r in sorted(drift, key=lambda r: (PERIOD_ORDER.index(r["period"]), FEATURE_ORDER.index(r["feature"])))
    ],
    hide_index=True,
    width="stretch",
)
