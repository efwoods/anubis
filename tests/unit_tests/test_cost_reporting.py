"""Tests for cost and usage reporting functionality."""

from __future__ import annotations

import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.anubis.utils.reporting import cost_metrics, google_sheets, feature_tracking, alerts


class TestCostMetrics:
    """Tests for cost metrics computation."""

    @pytest.mark.asyncio
    async def test_compute_cost_per_avatar_no_pool(self):
        """Test cost per avatar computation with no database pool."""
        result = await cost_metrics.compute_cost_per_avatar(None)
        assert result["total_avatars"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["cost_per_avatar_usd"] == 0.0
        assert result["by_avatar"] == []

    @pytest.mark.asyncio
    async def test_compute_average_cost_per_message_no_pool(self):
        """Test average cost per message with no database pool."""
        result = await cost_metrics.compute_average_cost_per_message(None)
        assert result["total_messages"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["avg_cost_per_message_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_average_cost_per_conversation_no_pool(self):
        """Test average cost per conversation with no database pool."""
        result = await cost_metrics.compute_average_cost_per_conversation(None)
        assert result["total_conversations"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["avg_cost_per_conversation_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_cost_per_new_user_no_pool(self):
        """Test cost per new user with no database pool."""
        result = await cost_metrics.compute_cost_per_new_user(None)
        assert result["new_users"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["cost_per_new_user_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_compute_all_cost_metrics(self):
        """Test computing all metrics at once."""
        result = await cost_metrics.compute_all_cost_metrics(None)
        assert "timestamp" in result
        assert "period_start" in result
        assert "period_end" in result
        assert result["cost_per_avatar"] == 0.0
        assert result["avg_cost_per_message"] == 0.0
        assert result["avg_cost_per_conversation"] == 0.0
        assert result["cost_per_new_user"] == 0.0


class TestGoogleSheets:
    """Tests for Google Sheets integration."""

    @pytest.mark.asyncio
    async def test_ensure_sheets_client_no_credentials(self):
        """Test sheets client creation with no credentials."""
        context = MagicMock()
        context.google_sheets_service_account_json = None
        context.google_oauth_client_id = None
        context.google_oauth_client_secret = None
        
        client = await google_sheets.ensure_sheets_client(context)
        assert client is None

    def test_extract_spreadsheet_id_from_url(self):
        """Test extracting spreadsheet ID from URL."""
        url = "https://docs.google.com/spreadsheets/d/1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA/edit?gid=0#gid=0"
        sheet_id = google_sheets.extract_spreadsheet_id(url)
        assert sheet_id == "1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA"

    def test_extract_spreadsheet_id_from_id(self):
        """Test extracting spreadsheet ID when already an ID."""
        sheet_id = "1_f5q4gJ3gU0ynwMGvZVNARGp_VYT5XE6hcXtokA-VPA"
        result = google_sheets.extract_spreadsheet_id(sheet_id)
        assert result == sheet_id

    @pytest.mark.asyncio
    async def test_read_sheet_range_no_client(self):
        """Test reading sheet with no client."""
        context = MagicMock()
        context.google_sheets_service_account_json = None
        context.google_oauth_client_id = None
        
        result = await google_sheets.read_sheet_range(context, "test_id", "Sheet1!A1:B2")
        assert result is None

    @pytest.mark.asyncio
    async def test_write_sheet_range_no_client(self):
        """Test writing sheet with no client."""
        context = MagicMock()
        context.google_sheets_service_account_json = None
        context.google_oauth_client_id = None
        
        result = await google_sheets.write_sheet_range(
            context, "test_id", "Sheet1!A1:B2", [["a", "b"]]
        )
        assert result is False


class TestFeatureTracking:
    """Tests for feature cost tracking."""

    @pytest.mark.asyncio
    async def test_compute_feature_cost_no_pool(self):
        """Test feature cost computation with no pool."""
        result = await feature_tracking.compute_feature_cost(None)
        assert result["commits"] == []
        assert result["total_cost_usd"] == 0.0
        assert result["total_tokens"] == 0
        assert result["duration_seconds"] == 0.0
        assert result["cost_per_hour"] == 0.0

    @pytest.mark.asyncio
    async def test_predict_feature_cost_no_pool(self):
        """Test feature cost prediction with no pool."""
        result = await feature_tracking.predict_feature_cost(None, 10.0)
        assert result["predicted_cost_usd"] == 0.0
        assert result["predicted_tokens"] == 0
        assert result["confidence"] == "low"
        assert result["basis"] == "no_data"

    def test_get_git_commits_since_invalid_repo(self):
        """Test getting git commits from invalid repo."""
        commits = feature_tracking.get_git_commits_since("/nonexistent")
        assert commits == []


class TestAlerts:
    """Tests for cost alerting."""

    @pytest.mark.asyncio
    async def test_check_cost_thresholds_no_threshold(self):
        """Test threshold check with no thresholds configured."""
        context = MagicMock()
        context.cost_alert_daily_threshold_usd = None
        context.cost_alert_monthly_threshold_usd = None
        
        alerts_list = await alerts.check_cost_thresholds(
            context, MagicMock(), "user_id", "assistant_id"
        )
        assert alerts_list == []

    @pytest.mark.asyncio
    async def test_detect_cost_anomalies_no_pool(self):
        """Test anomaly detection with no pool."""
        alerts_list = await alerts.detect_cost_anomalies(None, "user_id")
        assert alerts_list == []

    @pytest.mark.asyncio
    async def test_send_cost_alert_to_inbox(self):
        """Test sending alert to inbox."""
        context = MagicMock()
        pool = MagicMock()
        store = AsyncMock()
        store.aput = AsyncMock()
        
        alert = {
            "type": "test_alert",
            "subject": "Test Alert",
            "body": "Test body",
            "severity": "high",
        }
        
        result = await alerts.send_cost_alert_to_inbox(
            context, pool, store, "user_id", "assistant_id", alert
        )
        assert result is True
        store.aput.assert_called_once()


class TestProviderAdapters:
    """Tests for provider usage adapters."""

    @pytest.mark.asyncio
    async def test_fetch_openai_usage_no_key(self):
        """Test OpenAI usage fetch with no API key."""
        from src.anubis.utils.reporting.providers import openai_adapter
        
        context = MagicMock()
        context.openai_api_key = None
        
        result = await openai_adapter.fetch_openai_usage(context)
        assert result["total_cost_usd"] == 0.0
        assert result["total_tokens"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_elevenlabs_usage_no_key(self):
        """Test ElevenLabs usage fetch with no API key."""
        from src.anubis.utils.reporting.providers import elevenlabs_adapter
        
        context = MagicMock()
        context.elevenlabs_api_key = None
        
        result = await elevenlabs_adapter.fetch_elevenlabs_usage(context)
        assert result["total_cost_usd"] == 0.0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_fetch_xai_usage_no_key(self):
        """Test xAI usage fetch with no API key."""
        from src.anubis.utils.reporting.providers import xai_adapter
        
        context = MagicMock()
        context.xai_api_key = None
        
        result = await xai_adapter.fetch_xai_usage(context)
        assert result["total_cost_usd"] == 0.0
        assert "error" in result
