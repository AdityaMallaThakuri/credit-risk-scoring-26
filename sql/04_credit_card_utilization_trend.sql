-- Rolling credit utilization trend from credit_card_balance.
-- MONTHS_BALANCE verified <= -1 for all rows -- entirely historical
-- relative to the current application.

DROP TABLE IF EXISTS credit_card_utilization_trend;

CREATE TABLE credit_card_utilization_trend AS
WITH cc_calc AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        CASE WHEN AMT_CREDIT_LIMIT_ACTUAL > 0
             THEN CAST(AMT_BALANCE AS REAL) / AMT_CREDIT_LIMIT_ACTUAL
             ELSE NULL END AS utilization
    FROM credit_card_balance
),
cc_rolling AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        utilization,
        -- rolling 3-month average utilization per card, ordered chronologically
        AVG(utilization) OVER (
            PARTITION BY SK_ID_CURR, SK_ID_PREV
            ORDER BY MONTHS_BALANCE
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3mo_utilization,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR, SK_ID_PREV ORDER BY MONTHS_BALANCE DESC) AS rn_most_recent,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR, SK_ID_PREV ORDER BY MONTHS_BALANCE ASC) AS rn_earliest
    FROM cc_calc
),
cc_card_trend AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MAX(CASE WHEN rn_most_recent = 1 THEN rolling_3mo_utilization END) AS recent_utilization,
        MAX(CASE WHEN rn_earliest = 1 THEN rolling_3mo_utilization END) AS earliest_utilization,
        AVG(utilization) AS avg_utilization,
        COUNT(*) AS months_of_history
    FROM cc_rolling
    GROUP BY SK_ID_CURR, SK_ID_PREV
)
SELECT
    SK_ID_CURR,
    AVG(avg_utilization) AS cc_avg_utilization,
    -- positive = utilization climbing over time (distress signal); negative = improving
    AVG(recent_utilization - earliest_utilization) AS cc_utilization_trend,
    SUM(months_of_history) AS cc_months_of_history
FROM cc_card_trend
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX idx_cc_trend_sk_id_curr ON credit_card_utilization_trend (SK_ID_CURR);
