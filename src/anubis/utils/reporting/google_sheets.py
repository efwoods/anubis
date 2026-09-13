"""Google Sheets integration for cost/usage reporting.

Personal avatars can read and update configured Google Sheets with live
reporting values using either service account credentials or OAuth.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


async def ensure_sheets_client(context: Any) -> Any | None:
    """Initialize and return a Google Sheets API client.

    Uses service account credentials from GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON
    or falls back to OAuth credentials from GOOGLE_OAUTH_CLIENT_ID/SECRET.
    Returns None if credentials are not configured.
    """
    service_account_json = getattr(context, "google_sheets_service_account_json", None)
    
    if service_account_json:
        # Service account authentication (preferred for automation)
        try:
            from google.oauth2 import service_account  # noqa: PLC0415
            from googleapiclient.discovery import build  # noqa: PLC0415
            
            credentials = service_account.Credentials.from_service_account_info(
                service_account_json,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            service = await asyncio.to_thread(
                build, "sheets", "v4", credentials=credentials, cache_discovery=False
            )
            return service
        except Exception as service_error:  # noqa: BLE001
            logger.error("Failed to create Sheets client from service account: %s", service_error)
            return None
    
    # OAuth flow (requires user interaction, not ideal for automated reporting)
    client_id = getattr(context, "google_oauth_client_id", None)
    client_secret = getattr(context, "google_oauth_client_secret", None)
    
    if not client_id or not client_secret:
        logger.warning(
            "Google Sheets credentials not configured. Set GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON "
            "or GOOGLE_OAUTH_CLIENT_ID/SECRET."
        )
        return None
    
    logger.warning("OAuth-based Sheets access requires user authentication flow; service account preferred.")
    return None


async def read_sheet_range(
    context: Any,
    spreadsheet_id: str,
    range_name: str,
) -> list[list[Any]] | None:
    """Read values from a Google Sheet range.

    Args:
        context: GlobalContext with credentials
        spreadsheet_id: The spreadsheet ID (from URL)
        range_name: A1 notation range (e.g., "Sheet1!A1:D10")

    Returns:
        List of rows, each row is a list of cell values, or None on error.
    """
    service = await ensure_sheets_client(context)
    if not service:
        return None
    
    try:
        result = await asyncio.to_thread(
            service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name,
            ).execute
        )
        return result.get("values", [])
    except Exception as read_error:  # noqa: BLE001
        logger.error(
            "Failed to read sheet %s range %s: %s",
            spreadsheet_id,
            range_name,
            read_error,
        )
        return None


async def write_sheet_range(
    context: Any,
    spreadsheet_id: str,
    range_name: str,
    values: list[list[Any]],
    value_input_option: str = "USER_ENTERED",
) -> bool:
    """Write values to a Google Sheet range.

    Args:
        context: GlobalContext with credentials
        spreadsheet_id: The spreadsheet ID (from URL)
        range_name: A1 notation range (e.g., "Sheet1!A1:D10")
        values: List of rows to write
        value_input_option: How to interpret input ("RAW" or "USER_ENTERED")

    Returns:
        True if write succeeded, False otherwise.
    """
    service = await ensure_sheets_client(context)
    if not service:
        return False
    
    try:
        body = {"values": values}
        await asyncio.to_thread(
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                body=body,
            ).execute
        )
        return True
    except Exception as write_error:  # noqa: BLE001
        logger.error(
            "Failed to write sheet %s range %s: %s",
            spreadsheet_id,
            range_name,
            write_error,
        )
        return False


async def append_sheet_rows(
    context: Any,
    spreadsheet_id: str,
    range_name: str,
    values: list[list[Any]],
    value_input_option: str = "USER_ENTERED",
) -> bool:
    """Append rows to a Google Sheet.

    Args:
        context: GlobalContext with credentials
        spreadsheet_id: The spreadsheet ID (from URL)
        range_name: A1 notation range (e.g., "Sheet1!A:D")
        values: List of rows to append
        value_input_option: How to interpret input ("RAW" or "USER_ENTERED")

    Returns:
        True if append succeeded, False otherwise.
    """
    service = await ensure_sheets_client(context)
    if not service:
        return False
    
    try:
        body = {"values": values}
        await asyncio.to_thread(
            service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                body=body,
            ).execute
        )
        return True
    except Exception as append_error:  # noqa: BLE001
        logger.error(
            "Failed to append to sheet %s range %s: %s",
            spreadsheet_id,
            range_name,
            append_error,
        )
        return False


def extract_spreadsheet_id(url_or_id: str) -> str:
    """Extract spreadsheet ID from a Google Sheets URL or return as-is if already an ID."""
    if not url_or_id:
        return ""
    
    # Handle full URL: https://docs.google.com/spreadsheets/d/{id}/edit...
    if "docs.google.com/spreadsheets" in url_or_id:
        parts = url_or_id.split("/d/")
        if len(parts) > 1:
            sheet_id = parts[1].split("/")[0].split("?")[0].split("#")[0]
            return sheet_id
    
    # Assume it's already an ID
    return url_or_id.strip()


async def update_cost_report(
    context: Any,
    metrics: dict[str, Any],
) -> bool:
    """Update the cost reporting spreadsheet with current metrics.

    Args:
        context: GlobalContext with sheet configuration
        metrics: Dictionary of metrics to report (cost_per_avatar, avg_cost_per_message, etc.)

    Returns:
        True if update succeeded, False otherwise.
    """
    sheet_url = getattr(context, "cost_reporting_spreadsheet_url", None)
    if not sheet_url:
        logger.warning("Cost reporting spreadsheet URL not configured.")
        return False
    
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    timestamp = datetime.now(UTC).isoformat()
    
    # Prepare row: [timestamp, metric1, metric2, ...]
    row = [
        timestamp,
        metrics.get("cost_per_avatar", 0.0),
        metrics.get("avg_cost_per_message", 0.0),
        metrics.get("avg_cost_per_conversation", 0.0),
        metrics.get("cost_per_new_user", 0.0),
        metrics.get("total_spend_usd", 0.0),
        metrics.get("total_tokens", 0),
    ]
    
    # Append to a "Metrics" sheet
    range_name = getattr(context, "cost_reporting_range", "Metrics!A:G")
    
    return await append_sheet_rows(
        context,
        spreadsheet_id,
        range_name,
        [row],
    )
