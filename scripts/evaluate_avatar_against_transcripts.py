#!/usr/bin/env python
"""Ask an avatar the questions a real interviewer asked, and score the replies.

The upload pipeline stores, for every diarized recording, the real exchanges
of the conversation: what the other speaker said, and what the avatar's person
really answered (store namespace ``langsmith_factual_q_and_a``, held out of
adapter training). This script replays those questions to the avatar through
``POST /message/{assistant_id}`` and scores each generated reply two ways:

- **accuracy** against what the person really answered (a judge model's
  claim-by-claim verdicts, and the embedding similarity of the two replies);
- **verisimilitude** of the generated text (the stylometric distance from the
  avatar's direct-quote cloud that the message path already computes, and the
  style features of the generated replies against the real replies).

Every question's full record — question, real reply, generated reply, thread
id, usage, style analysis and scores — is written to ``traces.jsonl``. That
file is the trace of the run. **No trace is sent to LangSmith**: the replies
are requested with ``stream=false``, a path that never registers a LangSmith
run, and the judge is called with tracing switched off. The report states what
the same turns would cost on LangSmith.

Each question runs on a new thread with ``evaluation=true``, which skips the
learning observation, so replayed interviewer words never count as the
administrator's own words.

The store URI names ``host.docker.internal``, so the usual invocation is from
inside the development API container:

    # List the examples and the projected LangSmith cost; calls no model.
    docker exec anubis-dev-langgraph-api-dev-1 \\
        python scripts/evaluate_avatar_against_transcripts.py \\
        --assistant-id 59e29e65-1b93-40b0-b579-d68fcc9c3018 --dry-run

    # Run 20 examples (paid: one avatar reply and one judge call per example).
    docker exec -e EVALUATION_API_KEY=... anubis-dev-langgraph-api-dev-1 \\
        python scripts/evaluate_avatar_against_transcripts.py \\
        --assistant-id 59e29e65-1b93-40b0-b579-d68fcc9c3018 --limit 20

Output goes to ``data/evaluations/<assistant_id>/<run label>/`` (``data/`` is
ignored by git): ``examples.json``, ``traces.jsonl``, ``summary.json`` and
``report.md``. Comparing two runs' ``summary.json`` files is how a prompt or
data change is tuned.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, List

# Evaluation must never consume LangSmith traces. Set before any LangChain
# import reads the environment.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.context import GlobalContext  # noqa: E402
from src.anubis.utils.evaluation.transcript_evaluation import (  # noqa: E402
    QUESTION_AND_ANSWER_NAMESPACE,
    QUOTE_NAMESPACE,
    TRANSCRIPT_ANSWER_JUDGE_SYSTEM_PROMPT,
    EvaluationExample,
    EvaluationTrace,
    TranscriptAnswerJudgement,
    accuracy_score_from_judgement,
    build_evaluation_examples,
    compare_style_feature_means,
    cosine_similarity,
    judge_human_message,
    langsmith_trace_cost_projection,
    render_evaluation_report,
    select_evaluation_examples,
    summarize_evaluation,
    verisimilitude_score_from_response_metadata,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assistant-id", required=True)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("EVALUATION_API_BASE_URL", "http://localhost:9600"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EVALUATION_API_KEY"),
        help="API-KEY of the administrator account (or set EVALUATION_API_KEY).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", action="append", default=None,
                        help="Keep examples whose source contains this text; repeatable.")
    parser.add_argument("--minimum-question-words", type=int, default=4)
    parser.add_argument("--minimum-answer-words", type=int, default=12)
    parser.add_argument("--include-synthetic-questions", action="store_true",
                        help="Also keep examples whose question was generated, not really asked.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the judge model; accuracy is embedding similarity only.")
    parser.add_argument("--no-embedding", action="store_true",
                        help="Skip the local embedding similarity.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the examples and the cost projection; call no model.")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def read_stored_rows(store_uri: str, assistant_id: str) -> Dict[str, Any]:
    """Read the avatar's record, dataset rows and quote rows with plain SQL."""
    import psycopg

    with psycopg.connect(store_uri, autocommit=True, connect_timeout=15) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "select name, metadata->>'user_id' from assistant where assistant_id = %s",
                (assistant_id,),
            )
            assistant_row = cursor.fetchone()
            if assistant_row is None:
                raise SystemExit(f"No avatar with assistant_id {assistant_id}")
            avatar_name, owner_user_id = assistant_row
            prefix = f"{owner_user_id}.{assistant_id}."
            cursor.execute(
                "select value from store where prefix like %s",
                (f"{prefix}{QUESTION_AND_ANSWER_NAMESPACE}.%",),
            )
            question_and_answer_rows = [row[0] for row in cursor.fetchall()]
            cursor.execute(
                "select value->'document'->'kwargs'->>'page_content', "
                "value->'document'->'kwargs'->'metadata' "
                "from store where prefix like %s",
                (f"{prefix}{QUOTE_NAMESPACE}.%",),
            )
            quote_rows = [
                {"page_content": page_content, "metadata": metadata or {}}
                for page_content, metadata in cursor.fetchall()
            ]
    return {
        "avatar_name": avatar_name,
        "owner_user_id": owner_user_id,
        "question_and_answer_rows": question_and_answer_rows,
        "quote_rows": quote_rows,
    }


def sum_recorded_reply_cost(store_uri: str, thread_ids: List[str]) -> float | None:
    """Sum what ``api_metrics`` recorded for the run's threads.

    Each avatar reply writes the reply's own ``api_metrics`` row, so the cost of
    the run's replies is one SQL sum over the run's thread ids.
    """
    if not thread_ids:
        return None
    import psycopg

    with psycopg.connect(store_uri, autocommit=True, connect_timeout=15) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "select coalesce(sum(cost_usd), 0), count(*) from api_metrics "
                "where thread_id = any(%s)",
                (thread_ids,),
            )
            total, row_count = cursor.fetchone()
    return float(total) if row_count else None


async def ask_avatar(
    client: Any, arguments: argparse.Namespace, example: EvaluationExample
) -> Dict[str, Any]:
    """Ask the avatar one question on a new thread; return the response body."""
    started = time.monotonic()
    response = await client.post(
        f"{arguments.api_base_url}/message/{arguments.assistant_id}",
        headers={"API-KEY": arguments.api_key},
        data={
            "message": example.question,
            "stream": "false",
            "evaluation": "true",
            "include_quality_metrics": "true",
            "include_usage_metrics": "true",
            "conversation_title": f"evaluation {example.example_id}"[:120],
        },
    )
    response.raise_for_status()
    body = response.json()
    body["_client_latency_milliseconds"] = (time.monotonic() - started) * 1000.0
    return body


def reply_text(content: Any) -> str:
    """Flatten a reply's content (a string or a list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text") or "") if isinstance(block, dict) else str(block)
            for block in content
        ).strip()
    return str(content or "")


async def judge_reply(
    judge_model: Any, example: EvaluationExample, generated_answer: str
) -> tuple[TranscriptAnswerJudgement | None, Dict[str, Any]]:
    """Call the judge model once; return the judgement and the token usage."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await judge_model.ainvoke(
        [
            SystemMessage(content=TRANSCRIPT_ANSWER_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=judge_human_message(example, generated_answer)),
        ]
    )
    usage: Dict[str, Any] = {}
    judgement = response
    if isinstance(response, dict):
        judgement = response.get("parsed")
        raw = response.get("raw")
        usage = dict(getattr(raw, "usage_metadata", None) or {})
    if not isinstance(judgement, TranscriptAnswerJudgement):
        return None, usage
    return judgement, usage


def build_judge_model(context: GlobalContext) -> Any:
    """The classification model with structured output and raw usage kept."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=context.classification_model,
        api_key=context.classification_model_api_key or context.llm_provider_api_key,
        base_url=context.classification_model_base_url or context.llm_provider_base_url,
    )
    return model.with_structured_output(TranscriptAnswerJudgement, include_raw=True)


async def run_evaluation(arguments: argparse.Namespace) -> int:
    context = GlobalContext()
    store_uri = context.async_postgres_store_uri
    stored = read_stored_rows(store_uri, arguments.assistant_id)
    all_examples = build_evaluation_examples(
        stored["question_and_answer_rows"], stored["quote_rows"]
    )
    examples = select_evaluation_examples(
        all_examples,
        real_questions_only=not arguments.include_synthetic_questions,
        minimum_question_words=arguments.minimum_question_words,
        minimum_answer_words=arguments.minimum_answer_words,
        sources=arguments.source,
        limit=arguments.limit,
    )
    run_label = arguments.run_label or datetime.now(tz=UTC).strftime(
        "%Y-%m-%dT%H-%M-%SZ"
    )
    output_directory = os.path.join(
        os.path.dirname(__file__), "..", "data", "evaluations", arguments.assistant_id, run_label
    )
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, "examples.json"), "w") as examples_file:
        json.dump([asdict(example) for example in examples], examples_file, indent=2)

    real_question_count = sum(1 for example in all_examples if example.question_is_real)
    print(f"Avatar: {stored['avatar_name']} ({arguments.assistant_id})")
    print(
        f"Stored examples: {len(all_examples)} "
        f"({real_question_count} with a real question, "
        f"{len(all_examples) - real_question_count} with a generated question)"
    )
    print(f"Selected examples: {len(examples)}")
    by_source: Dict[str, int] = {}
    for example in examples:
        by_source[example.source] = by_source.get(example.source, 0) + 1
    for source, count in sorted(by_source.items(), key=lambda item: -item[1]):
        print(f"  {count:4d}  {source}")
    projection = langsmith_trace_cost_projection(len(examples))
    print(f"LangSmith cost if these turns were traced: {projection['arithmetic']}")

    if arguments.dry_run:
        print("Dry run: no model was called. Model cost: not measured.")
        print(f"Examples written to {os.path.abspath(output_directory)}/examples.json")
        return 0
    if not examples:
        raise SystemExit("No examples passed the selection; nothing to evaluate.")
    if not arguments.api_key:
        raise SystemExit("Set --api-key or EVALUATION_API_KEY (the administrator's API-KEY).")

    import httpx

    judge_model = None if arguments.no_judge else build_judge_model(context)
    embedding_model = None
    if not arguments.no_embedding:
        from sentence_transformers import SentenceTransformer

        embedding_model = await asyncio.to_thread(SentenceTransformer, context.embedding_model)

    from src.anubis.utils.billing.metering import price_model_token_usage
    from src.anubis.utils.dataset.style_features import extract_style_features

    semaphore = asyncio.Semaphore(max(1, arguments.concurrency))
    judge_cost_dollars = 0.0

    async def evaluate_one(client: Any, example: EvaluationExample) -> EvaluationTrace:
        nonlocal judge_cost_dollars
        async with semaphore:
            try:
                body = await ask_avatar(client, arguments, example)
            except Exception as request_error:  # noqa: BLE001 - one failure never ends the run
                return EvaluationTrace(
                    example=example,
                    generated_answer="",
                    thread_id=None,
                    latency_milliseconds=None,
                    reply_usage={},
                    judge_usage={},
                    accuracy=accuracy_score_from_judgement(None, semantic_similarity=None),
                    verisimilitude=verisimilitude_score_from_response_metadata(
                        None, generated_answer="", real_answer=example.real_answer
                    ),
                    error=f"{type(request_error).__name__}: {request_error}",
                )
            generated_answer = reply_text(body.get("content"))
            response_metadata = body.get("response_metadata") or {}

            semantic_similarity = None
            if embedding_model is not None and generated_answer:
                vectors = await asyncio.to_thread(
                    embedding_model.encode, [generated_answer, example.real_answer]
                )
                semantic_similarity = cosine_similarity(
                    [float(value) for value in vectors[0]],
                    [float(value) for value in vectors[1]],
                )

            judgement, judge_usage = None, {}
            if judge_model is not None and generated_answer:
                try:
                    judgement, judge_usage = await judge_reply(
                        judge_model, example, generated_answer
                    )
                    judge_cost_dollars += price_model_token_usage(
                        context.classification_model,
                        context,
                        prompt_tokens=int(judge_usage.get("input_tokens") or 0),
                        completion_tokens=int(judge_usage.get("output_tokens") or 0),
                    )
                except Exception as judge_error:  # noqa: BLE001
                    print(f"  judge failed for {example.example_id}: {judge_error}")

            trace = EvaluationTrace(
                example=example,
                generated_answer=generated_answer,
                thread_id=body.get("thread_id"),
                latency_milliseconds=float(
                    body.get("total_response_time_ms")
                    or body.get("_client_latency_milliseconds")
                    or 0.0
                ),
                reply_usage=body.get("usage") or {},
                judge_usage=judge_usage,
                accuracy=accuracy_score_from_judgement(
                    judgement, semantic_similarity=semantic_similarity
                ),
                verisimilitude=verisimilitude_score_from_response_metadata(
                    response_metadata,
                    generated_answer=generated_answer,
                    real_answer=example.real_answer,
                ),
                judgement=judgement.model_dump() if judgement is not None else None,
                response_metadata=response_metadata,
            )
            print(
                f"  {example.example_id}: precision="
                f"{trace.accuracy.claim_precision} similarity={semantic_similarity}"
            )
            return trace

    async with httpx.AsyncClient(timeout=arguments.request_timeout_seconds) as client:
        traces = list(
            await asyncio.gather(*[evaluate_one(client, example) for example in examples])
        )

    with open(os.path.join(output_directory, "traces.jsonl"), "w") as traces_file:
        for trace in traces:
            traces_file.write(json.dumps(trace.as_dictionary(), default=str) + "\n")

    answered = [trace for trace in traces if trace.error is None and trace.generated_answer]
    style_feature_comparison = compare_style_feature_means(
        [extract_style_features(trace.generated_answer) for trace in answered],
        [extract_style_features(trace.example.real_answer) for trace in answered],
    )
    reply_cost_dollars = sum_recorded_reply_cost(
        store_uri, [trace.thread_id for trace in traces if trace.thread_id]
    )
    summary = summarize_evaluation(
        traces,
        style_feature_comparison=style_feature_comparison,
        reply_cost_dollars=reply_cost_dollars,
        judge_cost_dollars=None if arguments.no_judge else judge_cost_dollars,
    )
    summary["run_label"] = run_label
    summary["assistant_id"] = arguments.assistant_id
    summary["thread_ids"] = [trace.thread_id for trace in traces if trace.thread_id]
    with open(os.path.join(output_directory, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2, default=str)
    report = render_evaluation_report(
        summary,
        traces,
        avatar_name=stored["avatar_name"],
        assistant_id=arguments.assistant_id,
        run_label=run_label,
    )
    with open(os.path.join(output_directory, "report.md"), "w") as report_file:
        report_file.write(report)
    print(f"Report: {os.path.abspath(output_directory)}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_evaluation(parse_arguments())))
