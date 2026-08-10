-- Aggregates the bureau + bureau_balance tables up to SK_ID_CURR.
-- All DAYS_CREDIT values are <= 0 (verified against raw data before writing
-- this: 0 / 1,716,428 positive), so every bureau record predates the current
-- application -- safe to aggregate in full, no temporal leakage.

DROP TABLE IF EXISTS bureau_features;

CREATE TABLE bureau_features AS
WITH bureau_recency AS (
    SELECT
        SK_ID_CURR,
        SK_ID_BUREAU,
        CREDIT_ACTIVE,
        DAYS_CREDIT,
        DAYS_CREDIT_ENDDATE,
        AMT_CREDIT_SUM,
        AMT_CREDIT_SUM_DEBT,
        AMT_CREDIT_SUM_OVERDUE,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_CREDIT DESC, SK_ID_BUREAU DESC) AS rn_most_recent
    FROM bureau
),
bureau_balance_delinquent AS (
    SELECT
        SK_ID_BUREAU,
        MONTHS_BALANCE
    FROM bureau_balance
    WHERE STATUS IN ('1', '2', '3', '4', '5')  -- 1-5 = increasing days-past-due buckets; C/X/0 excluded
),
bureau_balance_last_delinquency AS (
    -- MAX(MONTHS_BALANCE) = closest to 0 = most recent delinquent month, per bureau line
    SELECT SK_ID_BUREAU, MAX(MONTHS_BALANCE) AS months_balance_last_delinquent
    FROM bureau_balance_delinquent
    GROUP BY SK_ID_BUREAU
),
bureau_delinquency_per_curr AS (
    -- most recent delinquency across ALL of a client's bureau lines.
    -- Sign flipped to a positive "months since" distance (MAX(MONTHS_BALANCE)
    -- is the least-negative/most-recent value; negate so larger = longer ago).
    SELECT b.SK_ID_CURR, -MAX(bbld.months_balance_last_delinquent) AS bureau_months_since_last_delinquency
    FROM bureau b
    JOIN bureau_balance_last_delinquency bbld ON bbld.SK_ID_BUREAU = b.SK_ID_BUREAU
    GROUP BY b.SK_ID_CURR
),
bureau_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS bureau_total_credit_count,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS bureau_active_credit_count,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS bureau_closed_credit_count,
        AVG(DAYS_CREDIT) AS bureau_avg_days_credit,
        SUM(AMT_CREDIT_SUM) AS bureau_total_amt_credit_sum,
        SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_amt_credit_sum_debt,
        SUM(CASE WHEN AMT_CREDIT_SUM_OVERDUE > 0 THEN 1 ELSE 0 END) AS bureau_overdue_credit_count,
        -- DAYS_CREDIT_ENDDATE is a contractual/scheduled date set at origination (known at the
        -- time), so this term-length derivation does not use anything observed after the fact.
        -- Known data-quality quirk (checked against raw bureau.csv): 164/1,716,428 rows (0.01%)
        -- have DAYS_CREDIT_ENDDATE before DAYS_CREDIT, giving a negative term. Left unfiltered --
        -- negligible at this rate, and filtering would itself bias the handful of affected clients.
        AVG(DAYS_CREDIT_ENDDATE - DAYS_CREDIT) AS bureau_avg_credit_term_days
    FROM bureau
    GROUP BY SK_ID_CURR
),
bureau_most_recent AS (
    -- DAYS_CREDIT is <= 0 (verified); negate to a positive "days since" distance
    SELECT SK_ID_CURR, -DAYS_CREDIT AS bureau_days_since_last_credit
    FROM bureau_recency
    WHERE rn_most_recent = 1
)
SELECT
    ba.SK_ID_CURR,
    ba.bureau_total_credit_count,
    ba.bureau_active_credit_count,
    ba.bureau_closed_credit_count,
    ba.bureau_avg_days_credit,
    bmr.bureau_days_since_last_credit,
    ba.bureau_total_amt_credit_sum,
    ba.bureau_total_amt_credit_sum_debt,
    CASE WHEN ba.bureau_total_amt_credit_sum > 0
         THEN ba.bureau_total_amt_credit_sum_debt / ba.bureau_total_amt_credit_sum
         ELSE NULL END AS bureau_credit_utilization,
    ba.bureau_overdue_credit_count,
    ba.bureau_avg_credit_term_days,
    bdc.bureau_months_since_last_delinquency
FROM bureau_agg ba
LEFT JOIN bureau_most_recent bmr ON bmr.SK_ID_CURR = ba.SK_ID_CURR
LEFT JOIN bureau_delinquency_per_curr bdc ON bdc.SK_ID_CURR = ba.SK_ID_CURR;

CREATE UNIQUE INDEX idx_bureau_features_sk_id_curr ON bureau_features (SK_ID_CURR);
