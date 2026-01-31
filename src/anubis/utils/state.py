# src/anubis/utils/state.py

from __future__ import annotations

import operator 

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages # Built-in reducer

from src.anubis.utils.helper_functions import add_queries
from langchain_core.documents import Document 
class GlobalState(TypedDict):
    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
    
    messages: Annotated[list[AnyMessage], add_messages] # type: ignore # enables append/update

    queries: Annotated[list[str], add_queries] = field(
        default_factory=list,
        metadata={
            "description":"A list of search queries that the agent has generated during vectorstore retrieval"
        }
    )

    retrieved_docs: Annotated[list[Document], operator.add] = field(
        default_factory=list, 
        metadata={
            "description":"Populated by the vector store graph retriever. This is a list of documents for reference during chat."
        }
    )
