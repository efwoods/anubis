"""Dataset formatting helpers.

Responsibilities:

* Llama-style adapter training format (single-turn / multi-turn).
* LangSmith example dict format for evaluation datasets.
* Per-line synthetic question generation that lets a corpus of direct quotes,
  monologues, presentations, and tweets be turned into ``(user_question,
  assistant_quote)`` pairs without ever modifying the assistant content.

These helpers are intentionally pure / deterministic where possible so that
upstream callers (``process_adapter_documents`` and the eval pipeline) can run
them concurrently with ``asyncio.gather``.
"""

import asyncio
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.model import init_model


""" LLAMA 4 ADAPTER TRAINING FORMAT """


async def llm_single_turn_dataset(
    question_list: List[str], answer_list: List[str]
) -> List[dict]:
    """Single-turn ``{"messages": [...]}`` rows for adapter training."""
    single_turn_dataset = []
    for question, answer in zip(question_list, answer_list):
        turn = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        single_turn_dataset.append(turn)
    return single_turn_dataset


def llm_multiturn_dataset_one_conversation(
    question_list: List[str], answer_list: List[str]
) -> dict:
    """Multi-turn ``{"messages": [...]}`` row built from a single conversation.

    Returns a single conversation dict; callers compose a list of these for the
    full training corpus.
    """
    list_of_messages = []
    for question, answer in zip(question_list, answer_list):
        turn = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        list_of_messages += turn
    multi_turn_dataset = {"messages": list_of_messages}
    return multi_turn_dataset


""" LANGSMITH DATASET FORMAT """


async def langsmith_dataset(
    question_list: List[str],
    answer_list: List[str],
    dataset_source_filename: str,
) -> List[dict]:
    """LangSmith example dicts ``[{inputs, outputs, metadata}, ...]``."""
    examples: List[dict] = []
    for question, answer in zip(question_list, answer_list):
        examples.append(
            {
                "inputs": {"question": question},
                "outputs": {"answer": answer},
                "metadata": {"source": dataset_source_filename},
            }
        )
    return examples


""" SYNTHETIC QUESTION GENERATION (per-line driver) """


class _GeneratedQuestion(BaseModel):
    """Structured response for a single synthetic question."""

    question: str


class _GeneratedQuestionsList(BaseModel):
    """Structured response for a list of synthetic questions in lockstep."""

    question_list: List[str]


_PER_LINE_SYSTEM_PROMPT = (
    "You generate a single, succinct user prompt that the supplied assistant "
    "response would naturally answer. The user prompt must NOT restate the "
    "answer, must NOT add new facts, and must be phrased in second person "
    "('you', 'your'). Return only the question."
)

async def generate_question_for_message(message_str: str) -> str:
    """Generate a single user prompt for a single assistant response.

    Used as the per-line driver for tweets, quotes, monologues, and
    presentations so that ``(user_prompt, assistant_quote)`` pairs can be
    emitted without modifying the assistant content.
    """
    model = init_model(model_without_tools=False, response_format=_GeneratedQuestion)
    messages = [
        SystemMessage(content=_PER_LINE_SYSTEM_PROMPT),
        HumanMessage(content=message_str),
    ]
    response = await model.ainvoke(input=messages)
    return response.question


async def create_question_list(str_messages_list: List[str]) -> List[str]:
    """Generate one synthetic user prompt per assistant response, concurrently.

    Preserves the input order. Internally calls
    :func:`generate_question_for_message` per item via ``asyncio.gather`` so
    larger corpora process in parallel.
    """
    if not str_messages_list:
        return []
    tasks = [generate_question_for_message(s) for s in str_messages_list]
    return list(await asyncio.gather(*tasks))


""" CORPUS-LEVEL ADAPTER + LANGSMITH BUILDER """


async def build_adapter_and_langsmith_for_quotes(
    quotes: List[str],
    dataset_source_filename: str,
    prompts: Optional[List[Optional[str]]] = None,
) -> Tuple[List[dict], List[dict]]:
    """Build adapter single-turn rows and LangSmith examples from a quote list.

    Each quote is paired with a user prompt. When ``prompts[i]`` carries a
    real, non-empty prerequisite statement (e.g. the preceding non-target turn
    in a diarized dialogue), it is used verbatim as the question; otherwise a
    synthetic user prompt is produced by :func:`generate_question_for_message`.
    Passing ``prompts=None`` (the default) synthesizes a question for every
    quote, preserving the original behavior. A single contiguous monologue or
    presentation chunk is supported by passing it as ``[chunk_text]``.

    Returns ``(adapter_rows, langsmith_rows)`` where ``adapter_rows`` is a list
    of ``{"messages": [...]}`` dicts and ``langsmith_rows`` is a list of
    ``{"inputs", "outputs", "metadata"}`` dicts.
    """
    if not quotes:
        return [], []

    if prompts is None:
        prompts = [None] * len(quotes)
    elif len(prompts) < len(quotes):
        prompts = list(prompts) + [None] * (len(quotes) - len(prompts))

    # Synthesize a question only where no real prompt was supplied.
    missing_idx = [
        i for i in range(len(quotes)) if not (prompts[i] or "").strip()
    ]
    synthetic_for_missing = await create_question_list(
        [quotes[i] for i in missing_idx]
    )
    synth_iter = iter(synthetic_for_missing)
    questions: List[str] = []
    for i in range(len(quotes)):
        real = (prompts[i] or "").strip()
        questions.append(real if real else next(synth_iter))

    adapter_rows = await llm_single_turn_dataset(
        question_list=questions, answer_list=quotes
    )
    langsmith_rows = await langsmith_dataset(
        question_list=questions,
        answer_list=quotes,
        dataset_source_filename=dataset_source_filename,
    )
    return adapter_rows, langsmith_rows


async def build_langsmith_for_conversation(
    messages: List[dict],
    dataset_source_filename: str,
) -> List[dict]:
    """Derive LangSmith ``(question, answer)`` pairs from a role-converted chat.

    ``messages`` is the role-converted conversation produced for adapter
    training (``[{"role": "user"|"assistant", "content": str}, ...]``). Each
    assistant turn becomes one example whose question is the immediately
    preceding user turn; when an assistant turn has no preceding user turn the
    question is synthesized via :func:`generate_question_for_message`.
    """
    question_list: List[str] = []
    answer_list: List[str] = []
    pending_question: Optional[str] = None
    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_question = content
        elif role == "assistant":
            question_list.append(pending_question or "")
            answer_list.append(content)
            pending_question = None

    if not answer_list:
        return []

    # Synthesize where the assistant led with no preceding user turn.
    missing_idx = [i for i, q in enumerate(question_list) if not q.strip()]
    if missing_idx:
        synth = await create_question_list(
            [answer_list[i] for i in missing_idx]
        )
        for j, i in enumerate(missing_idx):
            question_list[i] = synth[j]

    return await langsmith_dataset(
        question_list=question_list,
        answer_list=answer_list,
        dataset_source_filename=dataset_source_filename,
    )
