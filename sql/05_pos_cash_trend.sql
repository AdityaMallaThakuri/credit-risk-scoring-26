-- Rolling "utilization" trend from POS_CASH_balance. This table has no
-- credit-limit field like credit_card_balance, so utilization is proxied
-- by instalment completion: 1 - (instalments remaining / total instalments),
-- i.e. how much of the loan has been drawn down. MONTHS_BALANCE verified
-- <= -1 for all rows.

DROP TABLE IF EXISTS pos_cash_trend;

CREATE TABLE pos_cash_trend AS
WITH pos_calc AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        SK_DPD,
        CASE WHEN CNT_INSTALMENT > 0
             THEN 1.0 - (CAST(CNT_INSTALMENT_FUTURE AS REAL) / CNT_INSTALMENT)
             ELSE NULL END AS completion_ratio
    FROM pos_cash_balance
),
pos_rolling AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        completion_ratio,
        SK_DPD,
        AVG(completion_ratio) OVER (
            PARTITION BY SK_ID_CURR, SK_ID_PREV
            ORDER BY MONTHS_BALANCE
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_3mo_completion,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR, SK_ID_PREV ORDER BY MONTHS_BALANCE DESC) AS rn_most_recent,
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR, SK_ID_PREV ORDER BY MONTHS_BALANCE ASC) AS rn_earliest
    FROM pos_calc
),
pos_card_trend AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MAX(CASE WHEN rn_most_recent = 1 THEN rolling_3mo_completion END) AS recent_completion,
        MAX(CASE WHEN rn_earliest = 1 THEN rolling_3mo_completion END) AS earliest_completion,
        AVG(SK_DPD) AS avg_dpd,
        COUNT(*) AS months_of_history
    FROM pos_rolling
    GROUP BY SK_ID_CURR, SK_ID_PREV
)
SELECT
    SK_ID_CURR,
    AVG(recent_completion - earliest_completion) AS pos_completion_trend,
    AVG(avg_dpd) AS pos_avg_dpd,
    SUM(months_of_history) AS pos_months_of_history
FROM pos_card_trend
GROUP BY SK_ID_CURR;

CREATE UNIQUE INDEX idx_pos_trend_sk_id_curr ON pos_cash_trend (SK_ID_CURR);
