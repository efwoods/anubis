"""Spreadsheet header mapping for the owner's reference forecast."""

from src.anubis.utils.analytics.reference_forecasts import (
    metric_name_from_header,
    rows_from_sheet_values,
)


def test_header_to_metric_name():
    assert metric_name_from_header("Expected Burn") == "expected_burn"
    assert metric_name_from_header("Cost / user") == "cost_user"


def test_rows_from_sheet_values_maps_headers_and_skips_non_numeric():
    snapshots = rows_from_sheet_values(
        [
            ["Period", "Expected Burn", "Notes"],
            ["2026-09", "$1,200.50", "plan"],
            ["2026-10", "not a number", "skip"],
        ],
        spreadsheet_id="sheet-1",
        sheet_title="Budget",
    )
    assert len(snapshots) == 1
    assert snapshots[0]["metric"] == "expected_burn"
    assert snapshots[0]["value"] == 1200.50
    assert snapshots[0]["period"] == "2026-09"
    assert snapshots[0]["spreadsheet_id"] == "sheet-1"
