# src/anubis/utils/context.py

"""Define the runtime context information for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated


from src.anubis.utils.prompts import system_prompts
from src.anubis.utils.prompts.subgraphs import vector_store_graph_prompts

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
    metadata: dict = field(
        default=None, 
        metadata={"description": "This is metadata that includes the user_id of the creator."}
    )

@dataclass 
class UserContext(IdentityContext):
    pass

@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    assistant_ctx: AssistantContext = field(default_factory=AssistantContext)
    user_ctx: UserContext = field(default_factory=UserContext)

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
    

    llm_provider_api_key: str = field(
        default=None,
        metadata={
            "description": "API key for llama models"
        },
    )

    llm_provider_base_url: str = field(
        default=None,
        metadata={
            "description": "base url for the llama model"
        }
    )

    model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only; text response and tool use for thought processing."
        },
    )
    
    image_model: str = field(
        default=None,
        metadata={
            "description": "Model Name Only; used without tools for image to text descriptions."
        },
    )

    image_model_api_key: str = field(
        default=None,
        metadata={
            "description": "API Key; used without tools for image to text descriptions."
        },
    )

    image_model_base_url: str = field(
        default=None,
        metadata={
            "description": "Base Url; used without tools for image to text descriptions."
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

    deployment: str = field(
        default= None,
        metadata={"description": "True for langsmith deployments to use autoconfiguration of store; disables functionality of api yet allows the graph to run for deployments."}
    )

    supabase_url: str = field(
        default=None, 
        metadata={"description": "url for user authentication"}
    )

    supabase_key: str = field(
        default=None, 
        metadata={"description": "api key for user authentication"}
    )

    admin_user_id: str = field(
        default=None,
        metadata={"description": "user_id to allow the creation of public avatars. Reserved for CEO."}
    )

    anonymous_user_id: str = field(
        default=None,
        metadata={"description": "user_id to allow the creation of public avatars. Reserved for anonymous users to store the creation of avatars in a cookie."}
    )

    anonymous_api_key: str = field(
        default=None,
        metadata={"description": "api key for anonymous user data analytics to monitor content."}
    )

    # stripe_publishable_key: str = field(
    #     default=None, 
    #     metadata={"description": "API key for interacting with the stripe API."}
    # )

    stripe_secret_key: str = field(
        default=None, 
        metadata={"description": "API key for interacting with the stripe API."}
    )

    stripe_product_id: str = field(
        default=None, 
        metadata={"description": "Neural Nexus API monthly subscription product id."}
    )

    stripe_payment_url: str = field(
        default=None, 
        metadata={"description": "Payment URL for subscriptions."}
    )

    model_provider: str = field(
        default=None, 
        metadata={"description": "Model inference provider."}
    )

    llama_api_key: str = field(
        default = None, 
        metadata={"description": "LLama developer api key."}
    )

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))