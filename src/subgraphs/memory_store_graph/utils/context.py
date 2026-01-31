"""Define the runtime context information for the agent."""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from src.subgraphs.memory_store_graph.utils import prompts


@dataclass(kw_only=True)
class Context:
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

    system_prompt: str = prompts.SYSTEM_PROMPT

    llama_api_base_url: str = ""
    llama_api_key: str = ""

    def __post_init__(self):
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
