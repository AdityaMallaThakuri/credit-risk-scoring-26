"""Phase 7 dashboard, screen 6: the integrated Policy Simulator (roadmap
7.2 #6 -- the centerpiece screen). One threshold slider drives:
  1. Approval rate
  2. Expected Loss / (backtest) profit
  3. Live fairness metrics (recomputed at the slider's threshold, not the
     fixed Phase 3 optimal-threshold numbers screen 4 uses)
  4. A sample applicant's Weighted Temporal SHAP explanation

Honest scoping note (see the caption near the SHAP panel): Weighted
Temporal SHAP explains the model's margin/PD, which does not depend on
where the decision threshold is drawn -- so its bars don't change with
the slider. What DOES update live is that applicant's approved/declined
badge and their contribution to portfolio EL, which is threshold-
dependent. Faking a threshold-dependency in the SHAP values themselves
would misrepresent what the method actually measures.
"""

import plotly.graph_objects as go
import streamlit as st

from api_client import (
    get_applicant_page,
    get_applicant_score,
    get_deployed_threshold,
    get_fairness_live,
    get_simulation,
    get_weighted_explanation,
)

BLUE = "#2a78d6"
RED = "#e34948"
ORANGE = "#eb6834"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
PERIODS = ["P0", "P1", "P2", "P3", "P4"]

st.set_page_config(page_title="Policy Simulator", page_icon="\U0001F39B️", layout="wide")
st.title("Policy Simulator")

try:
    deployed_threshold = get_deployed_threshold()
    default_id = get_applicant_page(limit=1)["sk_id_curr"][0]
except Exception as e:
    st.error(f"Could not reach the backend API. Is `uvicorn main:app` running? ({e})")
    st.stop()

threshold = st.slider(
    "Decision threshold (approve if calibrated PD ≤ threshold)",
    min_value=0.0,
    max_value=1.0,
    value=float(deployed_threshold),
    step=0.005,
    format="%.3f",
)
if abs(threshold - deployed_threshold) < 1e-9:
    st.caption(f"This matches the deployed threshold ({deployed_threshold:.3f}).")

try:
    sim = get_simulation(threshold)
except Exception as e:
    st.error(f"Could not reach the backend API. ({e})")
    st.stop()

st.subheader("1 & 2 — Approval rate and Expected Loss / profit")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Approval rate", f"{sim['approval_rate']:.1%}")
col2.metric("Default rate among approved", f"{sim['approved_default_rate']:.1%}")
col3.metric("Total expected loss (approved)", f"${sim['total_expected_loss']:,.0f}")
col4.metric("Backtest profit", f"${sim['profit']:,.0f}")

st.divider()

st.subheader("3 — Fairness at this threshold")
fairness_attribute = st.selectbox("Attribute", options=["age_band", "CODE_GENDER", "NAME_FAMILY_STATUS"], key="fairness_attr")
try:
    fairness = get_fairness_live(threshold, fairness_attribute)
except Exception as e:
    st.error(f"Could not reach the backend API. ({e})")
    st.stop()

col5, col6, col7 = st.columns(3)
col5.metric("DPD", f"{fairness['DPD']:.4f}")
col6.metric("EOD", f"{fairness['EOD']:.4f}")
col7.metric("Equalized Odds Diff", f"{fairness['EqualizedOddsDiff']:.4f}")
st.caption(
    "Recomputed live at the current slider threshold (not the fixed Phase 3 optimal-threshold "
    "numbers on the Fairness Over Time screen) -- move the slider toward 0 or 1 and watch these "
    "numbers converge toward parity, with disparity often peaking somewhere in the middle rather "
    "than rising monotonically."
)
if fairness["excluded_groups"]:
    st.warning(
        f"Excluded from the DPD/EOD/EqualizedOddsDiff spread above (n<500, too small to be "
        f"reliable): {', '.join(fairness['excluded_groups'])}."
    )

st.divider()

st.subheader("4 — Sample applicant: Weighted Temporal SHAP")
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    sk_id_curr = st.number_input("SK_ID_CURR", min_value=1, value=int(default_id), step=1)
with col_ctrl2:
    period = st.selectbox("Period (drift context for the weighting)", options=PERIODS, index=PERIODS.index("P3"))
with col_ctrl3:
    alpha = st.slider("α (1.0 = pure drift-weighted, 0.0 = pure cost-weighted)", 0.0, 1.0, 0.5, 0.05)

st.info(
    "Weighted Temporal SHAP explains the model's margin/PD -- it has no notion of a decision "
    "threshold, so its bars below do **not** move when you drag the threshold slider above. "
    "What the threshold slider *does* drive for this applicant is the approved/declined badge "
    "just below, since that's the one threshold-dependent fact about them."
)

try:
    score = get_applicant_score(sk_id_curr)
    weighted = get_weighted_explanation(sk_id_curr, period, alpha, top_k=8)
except Exception as e:
    st.error(f"Could not look up SK_ID_CURR {sk_id_curr}. ({e})")
    st.stop()

approved_at_slider = score["pd_calibrated"] <= threshold
decision_color = "green" if approved_at_slider else "red"
col8, col9 = st.columns(2)
col8.metric("Calibrated PD", f"{score['pd_calibrated']:.1%}")
col9.markdown(f"**Decision at slider threshold {threshold:.3f}:** :{decision_color}[{'APPROVED' if approved_at_slider else 'DECLINED'}]")


def _clean_name(name: str) -> str:
    return name.removeprefix("num__").removeprefix("cat__")


contributions = weighted["contributions"]
ordered = sorted(contributions, key=lambda c: -c["weighted_shap_value"])
base_value = weighted["base_value"]
raw_margin = weighted["raw_margin"]
shown_sum = sum(c["weighted_shap_value"] for c in ordered)
other = (raw_margin - base_value) - shown_sum

x_labels = ["Base value"] + [_clean_name(c["feature"]) for c in ordered] + ["All other features", "Weighted margin"]
y_values = [base_value] + [c["weighted_shap_value"] for c in ordered] + [other, 0]
measures = ["absolute"] + ["relative"] * (len(ordered) + 1) + ["total"]

fig = go.Figure(
    go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=measures,
        increasing=dict(marker_color=RED),
        decreasing=dict(marker_color=BLUE),
        totals=dict(marker_color=MUTED),
        connector=dict(line=dict(color=GRIDLINE)),
    )
)
fig.update_layout(
    yaxis_title="Weighted margin contribution",
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    margin=dict(t=10),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Weighted_SHAP = static SHAP × [α·w_drift + (1−α)·w_cost] for period {period}, rescaled to "
    f"exact additive consistency: base {base_value:.3f} + shown + other = "
    f"{base_value + shown_sum + other:.3f}, matching the raw margin {raw_margin:.3f} exactly."
)

st.dataframe(
    [
        {
            "Feature": _clean_name(c["feature"]),
            "Value": c["value"],
            "Static SHAP": round(c["static_shap_value"], 4),
            "Weight": round(c["weight"], 4),
            "Weighted SHAP": round(c["weighted_shap_value"], 4),
        }
        for c in ordered
    ],
    hide_index=True,
    width="stretch",
)
