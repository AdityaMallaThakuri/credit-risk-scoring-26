"""Pydantic response models for the Phase 7 API. Kept separate from the
routers so response shape is reviewable in one place."""

from pydantic import BaseModel


class ApplicantScore(BaseModel):
    sk_id_curr: int
    pd_raw: float
    pd_calibrated: float
    threshold: float
    approved: bool
    expected_loss: float


class FeatureContribution(BaseModel):
    feature: str
    value: float | None
    shap_value: float


class ApplicantExplanation(BaseModel):
    sk_id_curr: int
    base_value: float
    raw_margin: float
    pd_calibrated: float
    contributions: list[FeatureContribution]


class RiskTierRow(BaseModel):
    tier: str
    count: int
    share: float
    default_rate: float
    avg_pd: float
    avg_ead: float


class DemographicSegmentRow(BaseModel):
    attribute: str
    group: str
    count: int
    approval_rate: float
    default_rate: float
    reliable: bool


class FairnessSyntheticRow(BaseModel):
    period: str
    flip_rate: float
    DPD: float
    EOD: float
    EqualizedOddsDiff: float


class FairnessRealRow(BaseModel):
    attribute: str
    DPD: float
    EOD: float
    EqualizedOddsDiff: float


class IsolatedBiasEffectRow(BaseModel):
    period: str
    DPD_true: float
    DPD_biased: float
    F_FPR_true: float
    F_FPR_biased: float
    F_FPR_inflation: float
    EqualizedOddsDiff_true: float
    EqualizedOddsDiff_biased: float


class LiveFairness(BaseModel):
    attribute: str
    threshold: float
    DPD: float
    EOD: float
    EqualizedOddsDiff: float
    excluded_groups: list[str]


class WeightedFeatureContribution(BaseModel):
    feature: str
    value: float | None
    static_shap_value: float
    weight: float
    weighted_shap_value: float


class WeightedShapExplanation(BaseModel):
    sk_id_curr: int
    period: str
    alpha: float
    base_value: float
    raw_margin: float
    pd_calibrated: float
    contributions: list[WeightedFeatureContribution]


class TradeoffRow(BaseModel):
    strategy: str
    auc: float
    approval_rate: float
    profit: float
    el_approved: float
    DPD: float
    EOD: float
    EqualizedOddsDiff: float


class DriftRow(BaseModel):
    period: str
    feature: str
    psi: float
    js_divergence: float
    ks_statistic: float
    ks_pvalue: float


class ProfitSimulation(BaseModel):
    threshold: float
    profit: float
    approval_rate: float
    approved_default_rate: float
    total_expected_loss: float
