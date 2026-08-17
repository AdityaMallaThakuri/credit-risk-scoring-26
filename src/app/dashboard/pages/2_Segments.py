"""Phase 7 dashboard, screen 2: Portfolio Segments (roadmap 7.2 #2).
Breaks the portfolio down by predicted risk tier and by demographic
group -- reuses data already computed for Phase 3 scoring and Phase 4
fairness metrics (see src/app/state.py's segment_population and
src/app/routers/segments.py), so this page is presentation only, no new
modeling.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import get_demographic_segments, get_risk_tier_segments

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRIDLINE = "#e1e0d9"

# Status palette -- risk TIER is a state (good/warning/critical), not an
# identity, so this is the palette this represents, per the dataviz
# skill's rule that status colors are reserved for state.
STATUS_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
TIER_ORDER = ["Low", "Medium", "High"]

st.set_page_config(page_title="Portfolio Segments", page_icon="\U0001F9E9", layout="wide")
st.title("Portfolio Segments")

try:
    risk_tiers = get_risk_tier_segments()
except Exception as e:
    st.error(f"Could not reach the backend API. Is `uvicorn main:app` running? ({e})")
    st.stop()

risk_tiers = sorted(risk_tiers, key=lambda r: TIER_ORDER.index(r["tier"]))

st.subheader("By predicted risk tier")
st.caption(
    "Tier cutoffs: Low = PD < 0.10, Medium = 0.10 ≤ PD < 0.22, High = PD ≥ 0.22 "
    "(0.22 is the deployed decision threshold, so High = would be declined today). "
    "Approval rate is omitted here since it's a tautological 0%/100% split by "
    "construction of the tier boundaries -- default rate is the real signal."
)

col1, col2 = st.columns([1.3, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[r["tier"] for r in risk_tiers],
            y=[r["count"] for r in risk_tiers],
            marker_color=[STATUS_COLORS[r["tier"]] for r in risk_tiers],
            text=[f"{r['count']:,}  ({r['share']:.1%})" for r in risk_tiers],
            textposition="outside",
            width=0.5,
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Applicants",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        xaxis=dict(gridcolor=GRIDLINE, zeroline=False, categoryorder="array", categoryarray=TIER_ORDER),
        yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
        margin=dict(t=30),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    for r in risk_tiers:
        st.metric(
            f"{r['tier']} risk — observed default rate",
            f"{r['default_rate']:.1%}",
            help=f"avg PD {r['avg_pd']:.3f}, avg exposure ${r['avg_ead']:,.0f}",
        )

st.divider()

st.subheader("By demographic group")
attribute = st.selectbox(
    "Attribute",
    options=["CODE_GENDER", "NAME_FAMILY_STATUS", "age_band"],
    format_func=lambda a: {"CODE_GENDER": "Gender", "NAME_FAMILY_STATUS": "Family status", "age_band": "Age band"}[a],
)

try:
    demo_rows = get_demographic_segments(attribute)
except Exception as e:
    st.error(f"Could not reach the backend API. ({e})")
    st.stop()

if attribute == "age_band":
    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    demo_rows = sorted(demo_rows, key=lambda r: age_order.index(r["group"]))
else:
    demo_rows = sorted(demo_rows, key=lambda r: -r["count"])

reliable_rows = [r for r in demo_rows if r["reliable"]]
unreliable_rows = [r for r in demo_rows if not r["reliable"]]

groups = [r["group"] for r in reliable_rows]
fig2 = go.Figure()
fig2.add_trace(go.Bar(name="Approval rate", x=groups, y=[r["approval_rate"] for r in reliable_rows], marker_color=BLUE))
fig2.add_trace(go.Bar(name="Default rate", x=groups, y=[r["default_rate"] for r in reliable_rows], marker_color=ORANGE))
fig2.update_layout(
    barmode="group",
    bargap=0.3,
    yaxis_tickformat=".0%",
    yaxis_title="Rate",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    margin=dict(t=10),
)
st.plotly_chart(fig2, width="stretch")

if unreliable_rows:
    excluded_desc = ", ".join(f"{r['group']} (n={r['count']})" for r in unreliable_rows)
    st.warning(
        f"Excluded from the chart above -- too few applicants for a reliable rate (n<500): "
        f"{excluded_desc}. Their raw rate looks confident (e.g. a clean 100%/0%) but is just "
        f"noise from a handful of people, not a signal -- shown below for transparency, not "
        f"plotted alongside groups with real sample size."
    )
else:
    st.caption("Every group here has n≥500 -- no reliability concerns for this attribute.")

st.dataframe(
    [
        {
            "Group": r["group"],
            "Count": r["count"],
            "Approval rate": f"{r['approval_rate']:.1%}",
            "Default rate": f"{r['default_rate']:.1%}",
            "Reliable (n≥500)": "✓" if r["reliable"] else "⚠ low n",
        }
        for r in demo_rows
    ],
    hide_index=True,
    width="stretch",
)
