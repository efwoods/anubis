"""The SELECT-only gate for custom analytics SQL."""

from src.anubis.utils.analytics.sql_gate import (
    AnalyticsSqlRefused,
    validate_analytics_sql,
)
import pytest


def test_select_from_allowlisted_table_is_accepted():
    statement = validate_analytics_sql(
        "SELECT assistant_id, SUM(cost_usd) FROM api_metrics GROUP BY 1"
    )
    assert statement.startswith("SELECT")


def test_with_query_is_accepted():
    statement = validate_analytics_sql(
        "WITH days AS (SELECT created_at FROM api_metrics) SELECT * FROM days"
    )
    assert statement.upper().startswith("WITH")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM api_metrics",
        "SELECT * FROM api_metrics; DROP TABLE api_metrics",
        "SELECT * FROM store",
        "SELECT * FROM api_metrics --\n; DELETE FROM api_metrics",
        "INSERT INTO api_metrics (id) VALUES (1)",
        "SELECT 1",
        "",
    ],
)
def test_unsafe_or_unknown_sql_is_refused(sql):
    with pytest.raises(AnalyticsSqlRefused):
        validate_analytics_sql(sql)
