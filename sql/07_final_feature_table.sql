-- Final feature table: one row per SK_ID_CURR, LEFT JOINing all the
-- per-source aggregate tables onto application_train. LEFT JOIN (not
-- INNER) because not every applicant has bureau/previous-application/
-- credit-card/POS-cash history -- the has_*_history flags make "no
-- history" explicit rather than silently dropping those applicants.
-- TARGET is only ever joined in here, at the application_train level --
-- never into the auxiliary aggregation tables above (leakage-check item 4).

DROP TABLE IF EXISTS final_feature_table;

CREATE TABLE final_feature_table AS
SELECT
    a.SK_ID_CURR,
    a.TARGET,

    ats.WEEKDAY_APPR_PROCESS_START,
    ats.HOUR_APPR_PROCESS_START,
    ats.is_weekend_application,
    ats.is_night_application,

    bf.bureau_total_credit_count,
    bf.bureau_active_credit_count,
    bf.bureau_closed_credit_count,
    bf.bureau_avg_days_credit,
    bf.bureau_days_since_last_credit,
    bf.bureau_total_amt_credit_sum,
    bf.bureau_total_amt_credit_sum_debt,
    bf.bureau_credit_utilization,
    bf.bureau_overdue_credit_count,
    bf.bureau_avg_credit_term_days,
    bf.bureau_months_since_last_delinquency,

    pf.prev_app_count,
    pf.prev_app_approved_rate,
    pf.prev_app_refused_rate,
    pf.prev_app_avg_days_decision,
    pf.prev_app_avg_amt_credit,
    pf.prev_app_night_rate,
    pf.prev_app_weekend_rate,
    pf.prev_app_count_last_30d,
    pf.prev_app_count_last_90d,
    pf.prev_app_last_status,

    inf.installments_count,
    inf.installments_late_payment_rate,
    inf.installments_avg_days_late,
    inf.installments_avg_payment_ratio,
    inf.installments_avg_repayment_velocity,
    inf.installments_amt_volatility,
    inf.days_since_last_late_payment,

    cc.cc_avg_utilization,
    cc.cc_utilization_trend,
    cc.cc_months_of_history,

    pc.pos_completion_trend,
    pc.pos_avg_dpd,
    pc.pos_months_of_history,

    CASE WHEN bf.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS has_bureau_history,
    CASE WHEN pf.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS has_previous_application,
    CASE WHEN cc.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS has_credit_card_history,
    CASE WHEN pc.SK_ID_CURR IS NOT NULL THEN 1 ELSE 0 END AS has_pos_cash_history

FROM application_train a
LEFT JOIN application_time_signals ats ON ats.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN bureau_features bf ON bf.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN previous_application_features pf ON pf.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN installments_features inf ON inf.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN credit_card_utilization_trend cc ON cc.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN pos_cash_trend pc ON pc.SK_ID_CURR = a.SK_ID_CURR;

CREATE UNIQUE INDEX idx_final_feature_table_sk_id_curr ON final_feature_table (SK_ID_CURR);
