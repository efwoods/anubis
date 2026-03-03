# src/anubis/utils/configuration

"""Define the configurable parameters for the agent. Load Static Environment Variables."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated


import os

@dataclass(kw_only=True)
class GlobalConfiguration:
    """The configuration for the retrieval agent subgraph."""

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

