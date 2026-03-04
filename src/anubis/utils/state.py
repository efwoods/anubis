# src/anubis/utils/state.py

from __future__ import annotations

import operator 

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated


from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
from asyncio import Task
from langgraph.graph.message import add_messages # Built-in reducer
from langchain_core.documents import Document 


from src.anubis.utils.utility import (
    add_queries, 
    reduce_docs, 
)

import logging
logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class VectorstoreIndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed. Will be deleted after indexed
    """

    vectorstore_documents_to_be_indexed: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""

@dataclass(kw_only=True)
class AnalysisIndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed. Will be deleted after indexed
    """

    documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


@dataclass(kw_only=True)
class AdapterIndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed. Will be deleted after indexed
    """

    documents_to_be_processed_for_adapter_training: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""

@dataclass(kw_only=True)
class BaselineIndexState:
    """Represents the state for documents stored for textual baseline evaluation of generated content (stored in pg store).

    This class defines the structure of the index state, which includes
    the documents to be stored. Will be deleted after stored appropriately.
    """

    ground_truth_user_first_person_speech_baseline_for_evaluation: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can store or otherwise process."""


@dataclass(kw_only=True)
class UserIdentityState:
    """Represents the state for documents stored for textual baseline evaluation of generated content (stored in pg store).

    This class defines the structure of the index state, which includes
    the documents to be stored. Will be deleted after stored appropriately.
    """

    user_identity_documents: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can store or otherwise process."""

@dataclass(kw_only=True)
class AssistantIdentityState:
    """Represents the state for documents stored for textual baseline evaluation of generated content (stored in pg store).

    This class defines the structure of the index state, which includes
    the documents to be stored. Will be deleted after stored appropriately.
    """

    assistant_identity_documents: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can store or otherwise process."""

@dataclass(kw_only=True)
class RecalledMemories:
    """Represents the state of the memories that have been remembered and retrieved from the store.

    This class defines the structure of the recalled memories.
    """
    recalled_memory_documents: Annotated[Sequence[Document], reduce_docs] 

class AssistantState(TypedDict):
    assistant_id: str
    assistant_name: str
    assistant_identity: AssistantIdentityState
    recalled_memories: RecalledMemories

class UserState(TypedDict):
    user_id: str
    user_name: str
    user_identity: UserIdentityState

class GlobalState(TypedDict):
    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)


    messages: Annotated[list[AnyMessage], add_messages] # type: ignore # enables append/update

    assistant_state: AssistantState
    user_state: UserState

    """ Data Retrieval """

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

    recalled_memory_documents: Annotated[Sequence[Document], reduce_docs] 

    """ Data Uploading and Processing """

    # List of media items to be converted to text 

    media_files: Optional[List[Dict[str, Any]]] # Raw uploaded files

    media_list: Annotated[List[Dict], operator.add]  # media is moved into the task list and overwritten on message send from the chat message interface

    # List of media extracted from chat with the media type determined, and converted into text.
    processed_media_to_be_formatted: Annotated[Sequence[Document], operator.add]

    # List of Documents to be uploaded to the vectorstore (processed_media -> formatt -> vectorstore_documents)
    vectorstore_documents_to_be_indexed: VectorstoreIndexState

    # Analysis list
    documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant: AnalysisIndexState

    # Adapter list
    documents_to_be_processed_for_adapter_training: AdapterIndexState

    # Ground Truth User First Person Speech (literal quotes from the target entity)
    ground_truth_user_first_person_speech_baseline_for_evaluation: BaselineIndexState 

    """ Node Routing """

    route_decision: str = ""
