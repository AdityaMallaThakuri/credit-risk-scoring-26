-- Repayment velocity, delinquency recency, and payment volatility from
-- installments_payments. DAYS_INSTALMENT / DAYS_ENTRY_PAYMENT verified
-- <= 0 for all rows -- every instalment record predates the current
-- application.
-- SQRT is registered as a UDF by the Python driver (run_sql_pipeline.py) --
-- do not rely on SQLite's built-in math extension, which isn't guaranteed
-- to be compiled into every Python distribution's sqlite3 module.

DROP TABLE IF EXISTS installments_features;

CREATE TABLE installments_features AS
WITH installments_calc AS (
    SELECT
        SK_ID_CURR,
        NUM_INSTALMENT_NUMBER,
        DAYS_INSTALMENT,
        AMT_INSTALMENT,
        (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS days_late,  -- positive = paid after due date
        CASE WHEN AMT_INSTALMENT > 0 THEN AMT_PAYMENT / AMT_INSTALMENT ELSE NULL END AS payment_ratio
    FROM installments_payments
),
installments_late_recency AS (
    -- MAX(DAYS_INSTALMENT) = closest to 0 = most recent late instalment;
    -- negate to a positive "days since" distance
    SELECT SK_ID_CURR, -MAX(DAYS_INSTALMENT) AS days_since_last_late_payment
    FROM installments_calc
    WHERE days_late > 0
    GROUP BY SK_ID_CURR
),
installments_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*) AS installments_count,
        AVG(CASE WHEN days_late > 0 THEN 1.0 ELSE 0.0 END) AS installments_late_payment_rate,
        AVG(CASE WHEN days_late > 0 THEN days_late ELSE NULL END) AS installments_avg_days_late,
        AVG(payment_ratio) AS installments_avg_payment_ratio,
        AVG(-days_late) AS installments_avg_repayment_velocity,  -- higher (more negative days_late) = paid earlier
        AVG(AMT_INSTALMENT) AS installments_avg_amt,
        AVG(AMT_INSTALMENT * AMT_INSTALMENT) AS installments_avg_amt_sq
    FROM installments_calc
    GROUP BY SK_ID_CURR
)
SELECT
    ia.SK_ID_CURR,
    ia.installments_count,
    ia.installments_late_payment_rate,
    ia.installments_avg_days_late,
    ia.installments_avg_payment_ratio,
    ia.installments_avg_repayment_velocity,
    SQRT(MAX(ia.installments_avg_amt_sq - ia.installments_avg_amt * ia.installments_avg_amt, 0)) AS installments_amt_volatility,
    ilr.days_since_last_late_payment
FROM installments_agg ia
LEFT JOIN installments_late_recency ilr ON ilr.SK_ID_CURR = ia.SK_ID_CURR;

CREATE UNIQUE INDEX idx_installments_features_sk_id_curr ON installments_features (SK_ID_CURR);
