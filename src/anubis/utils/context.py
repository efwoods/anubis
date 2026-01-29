# src/agent/context.py

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

from src.anubis.utils import prompts

@dataclass(kw_only=True)
class Context:
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

    # def __post_init__(self) -> None:
    #     """Fetch env variables for attributes that were not passed as args"""
    #     for f in fields(self):
    #         if f not in f.init:
    #             continue

    #         if getattr(self, f.name) == f.default:
    #             setattr(self, f.name, os.environ.get(f.name.upper(), f.default))