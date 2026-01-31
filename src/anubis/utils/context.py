"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from src.anubis.utils import prompts

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from src.anubis.utils.prompts.system import prompts

@dataclass(kw_only=True)
class GlobalContext:
    """Main context class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""

    provider_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT, 
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

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))

@dataclass
class UserContext:
    user_id: str
    name: str
    description: str
    metadata: dict

@dataclass
class AssistantContext:
    user_id: str
    assistant_id: str
    name: str
    description: str
    metadata: dict