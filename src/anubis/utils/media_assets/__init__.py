"""Generated per-avatar media: emotion stills, idle loops, lip-sync clips, voice clips.

These assets are far larger than anything the LangGraph store should hold — a
six-second 720p clip is megabytes, and the store indexes every value through
the vector configuration — so they live in dedicated Postgres tables with
``BYTEA`` columns (:mod:`repository`), created on boot beside ``api_metrics``.

The repository is published process-wide the same way the connected-account
repository is (``set_media_asset_repository``), so the media-processing graph
can persist what it generates without importing the web application.
"""

from src.anubis.utils.media_assets.repository import (
    ASSET_KIND_IDLE_LOOP,
    ASSET_KIND_LIP_SYNC,
    ASSET_KIND_STILL,
    InMemoryMediaAssetRepository,
    PostgresMediaAssetRepository,
    ensure_media_asset_tables,
    get_media_asset_repository,
    set_media_asset_repository,
)

__all__ = [
    "ASSET_KIND_IDLE_LOOP",
    "ASSET_KIND_LIP_SYNC",
    "ASSET_KIND_STILL",
    "InMemoryMediaAssetRepository",
    "PostgresMediaAssetRepository",
    "ensure_media_asset_tables",
    "get_media_asset_repository",
    "set_media_asset_repository",
]
