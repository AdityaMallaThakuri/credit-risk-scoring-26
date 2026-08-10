-- Aggregates previous_application up to SK_ID_CURR, including
-- application-time behavioral signals computed from the client's PAST
-- applications' own WEEKDAY_APPR_PROCESS_START / HOUR_APPR_PROCESS_START.
-- DAYS_DECISION verified <= 0 for all rows (0 / 1,670,214 positive) --
-- every previous application predates the current one.

DROP TABLE IF EXISTS previous_application_features;

CREATE TABLE previous_application_features AS
WITH prev_recency AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        DAYS_DECISION,
        NAME_CONTRACT_STATUS,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC, SK_ID_PREV DESC) AS rn_most_recent
    FROM previous_application
),
prev_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS prev_app_count,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END) AS prev_app_approved_rate,
        AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END) AS prev_app_refused_rate,
        AVG(DAYS_DECISION) AS prev_app_avg_days_decision,
        AVG(AMT_CREDIT) AS prev_app_avg_amt_credit,
        AVG(CASE WHEN HOUR_APPR_PROCESS_START < 6 OR HOUR_APPR_PROCESS_START >= 22
                 THEN 1.0 ELSE 0.0 END) AS prev_app_night_rate,
        AVG(CASE WHEN WEEKDAY_APPR_PROCESS_START IN ('SATURDAY', 'SUNDAY')
                 THEN 1.0 ELSE 0.0 END) AS prev_app_weekend_rate,
        -- "number of applications in the last 30/90 days" -- desperation/fraud-adjacent signal
        SUM(CASE WHEN DAYS_DECISION >= -30 THEN 1 ELSE 0 END) AS prev_app_count_last_30d,
        SUM(CASE WHEN DAYS_DECISION >= -90 THEN 1 ELSE 0 END) AS prev_app_count_last_90d
    FROM previous_application
    GROUP BY SK_ID_CURR
),
prev_last_status AS (
    SELECT SK_ID_CURR, NAME_CONTRACT_STATUS AS prev_app_last_status
    FROM prev_recency
    WHERE rn_most_recent = 1
)
SELECT
    pa.SK_ID_CURR,
    pa.prev_app_count,
    pa.prev_app_approved_rate,
    pa.prev_app_refused_rate,
    pa.prev_app_avg_days_decision,
    pa.prev_app_avg_amt_credit,
    pa.prev_app_night_rate,
    pa.prev_app_weekend_rate,
    pa.prev_app_count_last_30d,
    pa.prev_app_count_last_90d,
    pls.prev_app_last_status
FROM prev_agg pa
LEFT JOIN prev_last_status pls ON pls.SK_ID_CURR = pa.SK_ID_CURR;

CREATE UNIQUE INDEX idx_prev_app_features_sk_id_curr ON previous_application_features (SK_ID_CURR);
