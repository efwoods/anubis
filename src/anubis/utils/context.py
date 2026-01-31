# src/anubis/utils/context.py

"""Define the runtime context information for the agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated


from src.anubis.utils.prompts import system_prompts
from src.anubis.utils.configuration import GlobalConfiguration

from langchain_core.messages import SystemMessage

@dataclass
class UserContext:
    user_id: str = field(default="default_user_id_1234")
    name: str = field(default=None)
    description: str = field(default=None)
    metadata: dict = field(default=None)

@dataclass
class AssistantContext:
    user_id: str = field(default="default_user_id_1234")
    assistant_id: str = field(default="Anubis") # Name of the Graph in langgraph.json
    name: str = field(default=None)
    description: str = field(default=None)
    metadata: dict = field(default=None)

@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    provider_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

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

    user_ctx: UserContext = field(default_factory=UserContext)
    assistant_ctx: AssistantContext = field(default_factory=AssistantContext)
    configuration: GlobalConfiguration = field(default_factory=GlobalConfiguration)

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
        self.configuration.__post_init__()

