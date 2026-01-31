# src/anubis/utils/configuration

"""Define the configurable parameters for the agent. Load Static Environment Variables."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config

from src.anubis.utils.prompts.subgraphs import vector_store_graph_prompts

import os

@dataclass(kw_only=True)
class IndexConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including user identification, embedding model selection,
    retriever provider choice, and search parameters.
    """

    user_id: str = field(default="test_user_1234", metadata={"description": "Unique identifier for the user."})
    # assistant_id: str = field(metadata={"description": "Unique identifier for the assistant."})

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["elastic", "elastic-local", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="mongodb",
        metadata={
            "description": "The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'mongodb'."
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: RunnableConfig | None = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=IndexConfiguration)


@dataclass(kw_only=True)
class GlobalConfiguration(IndexConfiguration):
    """The configuration for the retrieval agent subgraph."""

    response_system_prompt: str = field(
        default=vector_store_graph_prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="Llama-4-Maverick-17B-128E-Instruct-FP8",
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name."
        },
    )

    query_system_prompt: str = field(
        default=vector_store_graph_prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="Llama-4-Maverick-17B-128E-Instruct-FP8",
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name."
        },
    )

    """ Default Environment Variables """
    model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only"
        },
    )
    provider_model: str = field(
        default=None,
        metadata={
            "description": "provider/model_name"
        },
    )
    response_model: str = field(
        default=None,
        metadata={
            "description": "The system prompt used for processing and refining queries during vectorstore retrieval response."
        },
    )
    query_model: str = field(
        default=None,
        metadata={
            "description": "The system prompt used for processing and refining queries during query generation for the vectorstore."
        },
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

    mongodb_uri: str = field(
        default=None, 
        metadata={"description": "connection string to the mongodb for vectorstore retrieval."}
    )

    together_api_key: str = field(
        default=None, 
        metadata={"description": "inference provider for production use and for adapter training."}
    )


    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))

