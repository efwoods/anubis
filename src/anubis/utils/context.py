# src/anubis/utils/context.py

"""Define the runtime context information for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated


from src.anubis.utils.prompts import system_prompts
from src.anubis.utils.configuration import GlobalConfiguration

from langchain_core.messages import SystemMessage

from typing import Dict, Any

from langchain_postgres import PGVector
from langgraph.store.postgres import AsyncPostgresStore

from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine
import asyncpg

@dataclass
class IdentityContext:
    user_id: str = field(default="2feaa9d8-50c0-4550-81fa-9fb79bfe23f0")
    name: str = field(default=None)
    description: str = field(default=None)
    metadata: dict = field(default_factory=dict )

    def update_metadata(self, key: str, value: Any):
        """Update a specific metadata field."""
        self.metadata[key] = value
    
    def merge_metadata(self, new_metadata: Dict[str, Any]):
        """Merge new metadata into existing."""
        self._deep_merge(self.metadata, new_metadata)
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Recursively merge dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection."""
        return {
            "name": self.name,
            **self.metadata  # Unpack all metadata at top level
        }

@dataclass
class AssistantContext(IdentityContext):
    assistant_id: str = field(default="Anubis") # Name of the Graph in langgraph.json


@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    system_prompt: Annotated[SystemMessage, {"__template_metadata__": {"kind": "system_message"}}] = field(
        default=system_prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for interations."
            "Defines the context and behavior of the agent."
        },
    )

    max_search_results: int = field(
        default=10, 
        metadata={
            "description":"Maximum number of search results to return for each search query."
        },
    )

    llama_api_base_url: str = ""
    llama_api_key: str = ""

    temporary_system_prompt_update: str = ""


    user_ctx: IdentityContext = field(default_factory=IdentityContext)
    assistant_ctx: AssistantContext = field(default_factory=AssistantContext)
    configuration: GlobalConfiguration = field(default_factory=GlobalConfiguration)

    vector_store_memory_search_only: str = field(default="FALSE")

    async def load_identity_from_storage(self, user_id: str):
        """Load assistant identity from long-term storage."""
        # Simulate loading from database/vector store
        # In production, this would query your storage system

        stored_identity = {
            "name": "Elon Musk",
            "personality": {
                "traits": ["innovative", "direct", "ambitious"],
                "communication_style": "casual and technical"
            },
            "background": {
                "expertise": ["technology", "space", "electric vehicles"],
                "current_role": "CEO of multiple companies",
                "interests": ["Mars colonization", "AI safety", "sustainable energy"]
            },
            "knowledge_base": {
                "technical_depth": "expert",
                "fields": ["engineering", "physics", "business"]
            },
            "conversation_preferences": {
                "formality": "low",
                "humor": "dry and sarcastic",
                "detail_level": "high for technical topics"
            }
        }

        # Update assistant context
        self.assistant_ctx.name = stored_identity.get("name", None)
        self.assistant_ctx.merge_metadata(stored_identity)

    async def update_identity(self, new_information: Dict[str, Any]):
        """Update identity with new information during conversation."""
        self.assistant_ctx.merge_metadata(new_information)
        # Optionally persist to storage here
        await self._persist_to_storage()
    
    async def _persist_to_storage(self):
        """Persist updated identity to long-term storage."""
        # This would save to your database/vector store
        pass

    async def put_store_items(self, json):
        """Example put value to the store
        {
                  "namespace": [
                    ""
                  ],
                  "key": "",
                  "value": {}
                }

        Args:
            json (dict): put payload
        """

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))

        self.configuration.__post_init__()