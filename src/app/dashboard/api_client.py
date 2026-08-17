"""Thin HTTP client for the Streamlit dashboard against the Phase 7 FastAPI
backend (src/app/main.py). The dashboard is a separate process from the
API -- it only ever talks to it over HTTP, never imports src/app modules
directly, so it can be pointed at any deployment via API_BASE_URL."""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=60)
def get_applicant_page(limit: int = 1, offset: int = 0):
    r = requests.get(f"{API_BASE_URL}/applicants", params={"limit": limit, "offset": offset}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_applicant_score(sk_id_curr: int):
    r = requests.get(f"{API_BASE_URL}/applicants/{sk_id_curr}/score", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_applicant_explanation(sk_id_curr: int, top_k: int = 15):
    r = requests.get(f"{API_BASE_URL}/applicants/{sk_id_curr}/explanation", params={"top_k": top_k}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_simulation(threshold: float):
    r = requests.get(f"{API_BASE_URL}/simulate/threshold", params={"threshold": threshold}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_profit_curve():
    r = requests.get(f"{API_BASE_URL}/simulate/profit-curve", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_risk_tier_segments():
    r = requests.get(f"{API_BASE_URL}/segments/risk-tier", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_demographic_segments(attribute: str):
    r = requests.get(f"{API_BASE_URL}/segments/demographic", params={"attribute": attribute}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_fairness_by_period(period: str | None = None):
    params = {"period": period} if period else {}
    r = requests.get(f"{API_BASE_URL}/fairness/by-period", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_fairness_by_attribute(attribute: str | None = None):
    params = {"attribute": attribute} if attribute else {}
    r = requests.get(f"{API_BASE_URL}/fairness/by-attribute", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_fairness_live(threshold: float, attribute: str):
    r = requests.get(f"{API_BASE_URL}/fairness/live", params={"threshold": threshold, "attribute": attribute}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_weighted_explanation(sk_id_curr: int, period: str, alpha: float, top_k: int = 10):
    r = requests.get(
        f"{API_BASE_URL}/applicants/{sk_id_curr}/weighted-explanation",
        params={"period": period, "alpha": alpha, "top_k": top_k},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_fairness_isolated_effect():
    r = requests.get(f"{API_BASE_URL}/fairness/isolated-effect", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_fairness_tradeoff(attribute: str):
    r = requests.get(f"{API_BASE_URL}/fairness/tradeoff", params={"attribute": attribute}, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_drift(period: str | None = None, feature: str | None = None):
    params = {}
    if period:
        params["period"] = period
    if feature:
        params["feature"] = feature
    r = requests.get(f"{API_BASE_URL}/drift", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def get_deployed_threshold():
    """The joblib artifact's threshold is only exposed per-applicant-score
    response, not as its own endpoint -- read it off any one applicant
    rather than hardcoding a second copy of the same constant here."""
    sample_id = get_applicant_page(limit=1)["sk_id_curr"][0]
    return get_applicant_score(sample_id)["threshold"]
