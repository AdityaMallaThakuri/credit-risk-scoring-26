-- Application-time behavioral signals for the CURRENT application itself
-- (as opposed to 02_previous_application_features.sql, which computes the
-- same style of signal over the client's PAST applications).

DROP TABLE IF EXISTS application_time_signals;

CREATE TABLE application_time_signals AS
SELECT
    SK_ID_CURR,
    WEEKDAY_APPR_PROCESS_START,
    HOUR_APPR_PROCESS_START,
    CASE WHEN WEEKDAY_APPR_PROCESS_START IN ('SATURDAY', 'SUNDAY') THEN 1 ELSE 0 END AS is_weekend_application,
    CASE WHEN HOUR_APPR_PROCESS_START < 6 OR HOUR_APPR_PROCESS_START >= 22 THEN 1 ELSE 0 END AS is_night_application
FROM application_train;

CREATE UNIQUE INDEX idx_app_time_signals_sk_id_curr ON application_time_signals (SK_ID_CURR);
