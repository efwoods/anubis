# src/anubis/utils/state.py

from __future__ import annotations

import operator

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


from typing import Annotated, List, Dict, Optional, Any
from typing_extensions import TypedDict
from asyncio import Task
from langgraph.graph.message import add_messages  # Built-in reducer
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

    documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant: (
        Annotated[Sequence[Document], reduce_docs]
    )
    """A list of documents that the agent can index."""


@dataclass(kw_only=True)
class AdapterIndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed. Will be deleted after indexed
    """

    documents_to_be_processed_for_adapter_training: Annotated[
        Sequence[Document], reduce_docs
    ]
    """A list of documents that the agent can index."""


@dataclass(kw_only=True)
class BaselineIndexState:
    """Represents the state for documents stored for textual baseline evaluation of generated content (stored in pg store).

    This class defines the structure of the index state, which includes
    the documents to be stored. Will be deleted after stored appropriately.
    """

    ground_truth_user_first_person_speech_baseline_for_evaluation: Annotated[
        Sequence[Document], reduce_docs
    ]
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
    recalled_memory_documents: Annotated[Sequence[Document], reduce_docs]
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
    assistant_description: str


class UserState(TypedDict):
    user_id: str
    user_name: str
    user_description: str


from pydantic import BaseModel


class EmotionSummarization(BaseModel):
    """Given the retrieved content, summarize the feelings of the user. Detect current levels of emotion, emotional triggers, and clear reasoning."""

    emotional_summary: str
    emotional_summary_reasoning: str

    serenity: float
    joy: float
    ecstacy: float
    love: float
    acceptance: float
    trust: float
    admiration: float
    submission: float
    apprehension: float
    fear: float
    terror: float
    awe: float
    distraction: float
    surprise: float
    amazement: float
    disappointment: float
    pensiveness: float
    sadness: float
    grief: float
    remorse: float
    boredom: float
    disgust: float
    loathing: float
    contempt: float
    annoyance: float
    anger: float
    rage: float
    aggressiveness: float
    interest: float
    anticipation: float
    vigilance: float
    optimism: float


from langchain_core.messages import AIMessage, ToolMessage


class GlobalState(TypedDict):
    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)

    messages: Annotated[list[AnyMessage], add_messages]  # type: ignore # enables append/update

    internal_thoughts: Annotated[list[AIMessage, ToolMessage], add_messages]

    system_message: Annotated[list[SystemMessage], add_messages]

    """ Data Retrieval """

    queries: Annotated[list[str], add_queries] = field(
        default_factory=list,
        metadata={
            "description": "A list of search queries that the agent has generated during vectorstore retrieval"
        },
    )

    retrieved_docs: Annotated[list[Document], operator.add] = field(
        default_factory=list,
        metadata={
            "description": "Populated by the vector store graph retriever. This is a list of documents for reference during chat."
        },
    )

    """ User Identity"""
    user_state: UserState

    current_user_emotions: str
    # current_user_beliefs: str
    # current_user_desires: str
    # current_user_fears: str
    # current_user_hopes: str

    # current_user_action: str

    # current_user_objective: str

    user_identity_documents: Annotated[Sequence[Document], reduce_docs]

    """ Assistant Identity """
    assistant_state: AssistantState

    current_assistant_emotions: str
    # current_assistant_beliefs: str
    # current_assistant_desires: str
    # current_assistant_fears: str
    # current_assistant_hopes: str

    # current_assistant_action: str

    # current_user_objective: str

    recalled_memory_documents: Annotated[Sequence[Document], reduce_docs]
    assistant_identity_documents: Annotated[Sequence[Document], reduce_docs]

    """ Data Uploading and Processing """

    # List of media items to be converted to text

    media_files: Optional[List[Dict[str, Any]]]  # Raw uploaded files

    media_list: Annotated[
        List[Dict], operator.add
    ]  # media is moved into the task list and overwritten on message send from the chat message interface

    # List of media extracted from chat with the media type determined, and converted into text.
    processed_media_to_be_formatted: Annotated[Sequence[Document], operator.add]

    # List of Documents to be uploaded to the vectorstore (processed_media -> formatt -> vectorstore_documents)
    vectorstore_documents_to_be_indexed: VectorstoreIndexState

    # Analysis list
    documents_to_be_analyzed_for_context_storage_and_prompt_injection_of_assistant: (
        AnalysisIndexState
    )

    # Adapter list
    documents_to_be_processed_for_adapter_training: AdapterIndexState

    # Ground Truth User First Person Speech (literal quotes from the target entity)
    ground_truth_user_first_person_speech_baseline_for_evaluation: BaselineIndexState

    """ Node Routing """

    route_decision: str = ""

    """ Model Metrics """
    structured_response_calls_count: int = 0
    structured_response_prompt_tokens: int = 0
    structured_response_completion_tokens: int = 0
    structured_response_total_tokens: int = 0
    structured_response_total_cost: float = 0.0
    structured_response_latency_list_ms: Annotated[Sequence[float], operator.add]
    structured_response_average_latency_ms: float = 0.0

    image_model_calls_count: int = 0
    image_model_prompt_tokens: int = 0
    image_model_completion_tokens: int = 0
    image_model_total_tokens: int = 0
    image_model_total_cost: float = 0.0
    image_model_response_latency_list_ms: Annotated[Sequence[float], operator.add]
    image_model_average_latency_ms: float = 0.0

    inference_calls_count: int = 0
    inference_model_prompt_tokens: int = 0
    inference_model_completion_tokens: int = 0
    inference_model_total_tokens: int = 0
    inference_model_total_cost: float = 0.0
    inference_model_latency_list_ms: Annotated[Sequence[float], operator.add]
    inference_model_average_latency_ms: float = 0.0
