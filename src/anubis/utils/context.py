# src/anubis/utils/context.py

"""Define the runtime context information for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated


from src.anubis.utils.prompts import system_prompts
from src.anubis.utils.prompts.subgraphs import vector_store_graph_prompts

from src.anubis.utils.configuration import GlobalConfiguration

from langchain_core.messages import SystemMessage

from typing import Dict, Any

@dataclass
class IdentityContext:
    name: str = field(default=None)
    description: str = field(default=None)

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
    pass

@dataclass 
class UserContext(IdentityContext):
    pass

@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    user_ctx: UserContext = field(default_factory=UserContext)
    assistant_ctx: AssistantContext = field(default_factory=AssistantContext)

    # max_search_results: int = field(
    #     default=10, 
    #     metadata={
    #         "description":"Maximum number of search results to return for each search query."
    #     },
    # )

    # response_system_prompt: str = field(
    #     default=vector_store_graph_prompts.RESPONSE_SYSTEM_PROMPT,
    #     metadata={"description": "The system prompt used for generating responses."},
    # )

    # query_system_prompt: str = field(
    #     default=vector_store_graph_prompts.QUERY_SYSTEM_PROMPT,
    #     metadata={
    #         "description": "The system prompt used for processing and refining queries."
    #     },
    # )

    """ Default Environment Variables """

    together_api_key: str = field(
        default=None, 
        metadata={"description": "inference provider for production use and for adapter training."}
    )
    

    llama_api_key: str = field(
        default=None,
        metadata={
            "description": "API key for llama models"
        },
    )

    llama_api_base_url: str = field(
        default=None,
        metadata={
            "description": "base url for the llama model"
        }
    )

    model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only"
        },
    )

    dev: str = field(
        default=None,
        metadata={
            "description": "development mode; single user model; 10 requests/minute; no adapters/training"
        }
    )
    
    debug: str = field(
        default=None, 
        metadata={
            "description": "debugging available"
        }
    )

    huggingface_token: str = field(
        default=None,
        metadata={"description": "Token to use huggingface models"}
    )

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    vectorstore_postgres_uri: str = field(
        default=None,
        metadata={"description": "Connection string to postgres db for persistent document storage via vector store"}
    )

    async_postgres_store_uri: str = field(
        default=None,
        metadata={"description": "Connection string to async postgres store for persistent storage of avatar metadata for contextual prompt injection"}
    )

    model_token_limit: int = field(
        default=128000,
        metadata={"description": "number of acceptable tokens in a request to the current llm in thousands of tokens."}
    )

    langsmith_api_key: str = field(
        default = None, 
        metadata={"description": "api key"}
    )

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))