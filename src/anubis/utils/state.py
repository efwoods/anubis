from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage

from langgraph.graph import add_messages
from langgraph.graph.message import add_messages # Built-in reducer


class AnubisState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # enables append/update
