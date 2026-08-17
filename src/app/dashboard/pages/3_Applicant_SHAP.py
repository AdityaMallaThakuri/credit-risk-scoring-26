"""Phase 7 dashboard, screen 3: Individual Applicant view with local SHAP
explanation in plain language (roadmap 7.2 #3). Pure presentation on top
of the already-built /applicants/{id}/score and /applicants/{id}/explanation
endpoints (src/app/routers/scoring.py, explain.py) -- no new backend work
needed for this screen.
"""

import requests
import plotly.graph_objects as go
import streamlit as st

from api_client import get_applicant_explanation, get_applicant_page, get_applicant_score

BLUE = "#2a78d6"    # decreases predicted risk -- diverging pair, cool pole
RED = "#e34948"      # increases predicted risk -- diverging pair, warm pole
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
TOP_K = 8

st.set_page_config(page_title="Applicant Explanation", page_icon="\U0001F50E", layout="wide")
st.title("Individual Applicant — Score & Explanation")

try:
    first_page = get_applicant_page(limit=1)
    default_id = first_page["sk_id_curr"][0]
except Exception as e:
    st.error(f"Could not reach the backend API. Is `uvicorn main:app` running? ({e})")
    st.stop()

with st.expander("Browse applicant IDs"):
    offset = st.number_input("Offset", min_value=0, value=0, step=20)
    page = get_applicant_page(limit=20, offset=offset)
    st.write(f"Showing {offset + 1}–{offset + len(page['sk_id_curr'])} of {page['total']:,}")
    st.code(", ".join(str(i) for i in page["sk_id_curr"]))

sk_id_curr = st.number_input("SK_ID_CURR", min_value=1, value=int(default_id), step=1)

try:
    score = get_applicant_score(sk_id_curr)
    explanation = get_applicant_explanation(sk_id_curr, top_k=TOP_K)
except requests.HTTPError as e:
    if e.response is not None and e.response.status_code == 404:
        st.warning(f"SK_ID_CURR {sk_id_curr} not found in `modeling_feature_set`. Try an ID from the browser above.")
    else:
        st.error(f"API error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Could not reach the backend API. ({e})")
    st.stop()

decision = "APPROVED" if score["approved"] else "DECLINED"
decision_color = "green" if score["approved"] else "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Calibrated PD", f"{score['pd_calibrated']:.1%}")
col2.metric("Decision threshold", f"{score['threshold']:.1%}")
col3.markdown(f"**Decision:** :{decision_color}[{decision}]")
col4.metric("Expected loss", f"${score['expected_loss']:,.0f}")

st.divider()
st.subheader("Why the model scored this applicant this way")

contributions = explanation["contributions"]
base_value = explanation["base_value"]
raw_margin = explanation["raw_margin"]
shown_sum = sum(c["shap_value"] for c in contributions)
other = (raw_margin - base_value) - shown_sum


def _clean_name(name: str) -> str:
    return name.removeprefix("num__").removeprefix("cat__")


ordered = sorted(contributions, key=lambda c: -c["shap_value"])  # largest risk-increasing first

x_labels = ["Base value"] + [_clean_name(c["feature"]) for c in ordered] + ["All other features", "Final score (raw margin)"]
y_values = [base_value] + [c["shap_value"] for c in ordered] + [other, 0]
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
    yaxis_title="Model margin (higher = more risk)",
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    xaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    yaxis=dict(gridcolor=GRIDLINE, zeroline=False),
    margin=dict(t=10),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Base value {base_value:.3f} + top {TOP_K} contributions + all other features = "
    f"{base_value + shown_sum + other:.3f}, matching the raw margin {raw_margin:.3f} exactly "
    "(SHAP additive consistency, asserted server-side)."
)

st.subheader("In plain language")
top_up = max(contributions, key=lambda c: c["shap_value"])
top_down = min(contributions, key=lambda c: c["shap_value"])


def _describe(c):
    name = _clean_name(c["feature"])
    val = f"{c['value']:g}" if c["value"] is not None else "(categorical)"
    return f"**{name}** = {val}"


lines = [
    f"This applicant's calibrated default probability is **{score['pd_calibrated']:.1%}**, "
    f"against a decision threshold of {score['threshold']:.1%} — the loan is **{decision.lower()}**.",
]
if top_up["shap_value"] > 0:
    lines.append(f"The strongest factor **increasing** predicted risk is {_describe(top_up)} (contributes +{top_up['shap_value']:.3f} to the score).")
if top_down["shap_value"] < 0:
    lines.append(f"The strongest factor **decreasing** predicted risk is {_describe(top_down)} (contributes {top_down['shap_value']:.3f} to the score).")

st.markdown("\n\n".join(lines))

st.divider()
st.subheader(f"Top {TOP_K} feature contributions")
st.dataframe(
    [{"Feature": _clean_name(c["feature"]), "Value": c["value"], "SHAP contribution": round(c["shap_value"], 4)} for c in ordered],
    hide_index=True,
    width="stretch",
)
