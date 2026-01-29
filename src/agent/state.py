from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

@dataclass
class InputState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    

@dataclass
class State(InputState):
    # last step before the graph raises an error; 
    # 'True' when the step count reaches recursion_limit - 1.
    
    is_last_step: IsLastStep = field(default=False)

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)
