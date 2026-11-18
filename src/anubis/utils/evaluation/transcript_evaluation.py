"""Score an avatar's replies against the real person's replies in a transcript.

The upload pipeline already stores, per source recording, a question and
answer dataset under the store namespace ``langsmith_factual_q_and_a``: each
example pairs what an interviewer really said with what the avatar's person
really answered. That dataset is held out of adapter training. This module
turns the stored dataset into evaluation examples, scores one generated reply
against the real reply, and summarizes a run.

Two things are scored for every example.

**Accuracy with respect to the conversation.** Does the generated reply say
what the person really said in reply to that question? Measured by a judge
model that splits the generated reply into claims and marks each claim as
supported by, contradicted by, or absent from the real reply; and, at no model
cost, by the cosine similarity of the two replies' embeddings.

**Verisimilitude of the generated text.** Does the generated reply read like
the person's own speech, whatever the reply says? Measured by the stylometric
distance the message path already computes for every reply (squared
Mahalanobis distance of the reply's 28 style features from the avatar's
direct-quote cloud, against the avatar's empirical threshold), and by
comparing the style features of the generated replies with the style features
of the real replies to the same questions.

Everything here is pure: reading the store, calling the avatar and calling the
judge are done by ``scripts/evaluate_avatar_against_transcripts.py``. No part
of an evaluation sends a trace to LangSmith.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Sequence

from pydantic import BaseModel

QUESTION_AND_ANSWER_NAMESPACE = "langsmith_factual_q_and_a"
QUOTE_NAMESPACE = "quote"

# Measured on the LangSmith invoice GEHMVWPY0003 dated 2026-10-03: $182.58 for
# 46,515 base traces, and 46,515 traces for 6,704 graph invocations.
LANGSMITH_INVOICE_DOLLARS = 182.58
LANGSMITH_INVOICE_BASE_TRACES = 46515
LANGSMITH_INVOICE_GRAPH_INVOCATIONS = 6704


@dataclass
class EvaluationExample:
    """One real exchange: what was asked and what the person really answered."""

    example_id: str
    source: str
    question: str
    real_answer: str
    question_is_real: bool
    start_seconds: float | None = None
    end_seconds: float | None = None


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def word_count(text: str) -> int:
    return len((text or "").split())


def build_evaluation_examples(
    question_and_answer_rows: Iterable[Dict[str, Any]],
    quote_rows: Iterable[Dict[str, Any]],
) -> List[EvaluationExample]:
    """Build examples from the stored dataset rows and the stored quote rows.

    ``question_and_answer_rows`` are the store values of the
    ``langsmith_factual_q_and_a`` namespace: each has ``value`` (a JSON list of
    ``{"inputs": {"question"}, "outputs": {"answer"}, "metadata": {"source"}}``)
    and ``source_filename``.

    ``quote_rows`` are ``{"page_content", "metadata"}`` dictionaries of the
    ``quote`` namespace. A quote row whose ``adapter_prompt`` metadata is not
    empty records what the other speaker really said immediately before the
    quote. The dataset builder writes a synthetic question when no real one
    exists, so an example is marked ``question_is_real`` only when a quote row
    with the same source and the same answer text carries that same question
    as ``adapter_prompt``. The quote row also supplies the timestamps.
    """
    real_prompts: Dict[tuple, Dict[str, Any]] = {}
    for quote_row in quote_rows:
        metadata = quote_row.get("metadata") or {}
        adapter_prompt = (metadata.get("adapter_prompt") or "").strip()
        if not adapter_prompt:
            continue
        key = (
            str(metadata.get("source") or metadata.get("filename") or ""),
            _normalized_text(quote_row.get("page_content") or ""),
        )
        real_prompts[key] = {
            "adapter_prompt": adapter_prompt,
            "start": metadata.get("start"),
            "end": metadata.get("end"),
        }

    examples: List[EvaluationExample] = []
    for row in question_and_answer_rows:
        raw_examples = row.get("value")
        if isinstance(raw_examples, str):
            try:
                raw_examples = json.loads(raw_examples)
            except json.JSONDecodeError:
                continue
        row_source = str(row.get("source_filename") or "")
        for index, raw_example in enumerate(raw_examples or []):
            question = str((raw_example.get("inputs") or {}).get("question") or "").strip()
            answer = str((raw_example.get("outputs") or {}).get("answer") or "").strip()
            source = str((raw_example.get("metadata") or {}).get("source") or row_source)
            if not question or not answer:
                continue
            quote_match = real_prompts.get((source, _normalized_text(answer)))
            question_is_real = bool(
                quote_match
                and _normalized_text(quote_match["adapter_prompt"])
                == _normalized_text(question)
            )
            examples.append(
                EvaluationExample(
                    example_id=f"{source}#{index}",
                    source=source,
                    question=question,
                    real_answer=answer,
                    question_is_real=question_is_real,
                    start_seconds=_optional_float((quote_match or {}).get("start")),
                    end_seconds=_optional_float((quote_match or {}).get("end")),
                )
            )
    return examples


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def select_evaluation_examples(
    examples: Sequence[EvaluationExample],
    *,
    real_questions_only: bool = True,
    minimum_question_words: int = 4,
    minimum_answer_words: int = 12,
    sources: Sequence[str] | None = None,
    limit: int | None = None,
) -> List[EvaluationExample]:
    """Keep the examples that are a real question followed by a real answer.

    An interview is being reconstructed, so by default only examples whose
    question was really asked are kept. A one-word interjection is not a
    question and "Yeah." is not an answer worth scoring, so both sides have a
    minimum word count. ``sources`` keeps examples whose source contains any
    of the given substrings. ``limit`` keeps the first ``limit`` examples
    after an even spread across the sources, so one long recording cannot
    crowd out the others.
    """
    kept = [
        example
        for example in examples
        if (example.question_is_real or not real_questions_only)
        and word_count(example.question) >= minimum_question_words
        and word_count(example.real_answer) >= minimum_answer_words
        and (not sources or any(wanted in example.source for wanted in sources))
    ]
    if limit is None or len(kept) <= limit:
        return kept
    by_source: Dict[str, List[EvaluationExample]] = {}
    for example in kept:
        by_source.setdefault(example.source, []).append(example)
    spread: List[EvaluationExample] = []
    position = 0
    while len(spread) < limit:
        added = False
        for source_examples in by_source.values():
            if position < len(source_examples) and len(spread) < limit:
                spread.append(source_examples[position])
                added = True
        if not added:
            break
        position += 1
    return spread


# ---------------------------------------------------------------------------
# Accuracy: the judge
# ---------------------------------------------------------------------------


class JudgedClaim(BaseModel):
    """One claim made by the generated reply, judged against the real reply."""

    claim: str
    verdict: Literal["supported", "contradicted", "absent"]


class TranscriptAnswerJudgement(BaseModel):
    """The judge's reading of one generated reply against the real reply."""

    addresses_the_question: bool
    claims: List[JudgedClaim]
    real_reply_points_missing_from_generated_reply: List[str]
    explanation: str


TRANSCRIPT_ANSWER_JUDGE_SYSTEM_PROMPT = """<role>
You compare two replies to the same question. The real reply is what a person
really said in a recorded conversation. The generated reply is what an
artificial avatar of that person said when the avatar was asked the same
question. You decide how accurately the generated reply reproduces the content
of the real reply.
</role>

<task>
Read the human message. The human message gives the question, the real reply,
and the generated reply. Return exactly one `TranscriptAnswerJudgement`:

  addresses_the_question   True when the generated reply answers the question
                           that was asked; False when the generated reply
                           deflects, refuses, or answers a different question.
  claims                   Every distinct factual or opinion claim that the
                           generated reply makes, each with one verdict:
                             "supported"    the real reply states the same
                                            claim, in any wording.
                             "contradicted" the real reply states the opposite
                                            of the claim, or states a
                                            different value for the same fact.
                             "absent"       the real reply says nothing about
                                            the claim.
  real_reply_points_missing_from_generated_reply
                           Every distinct point of the real reply that the
                           generated reply does not make.
  explanation              Two sentences at most.
</task>

<rules>
- Judge content only. Wording, tone, length, filler words, and transcription
  errors in the real reply are not evidence for or against a claim.
- Split the generated reply into the smallest claims that can each be true or
  false alone. A greeting, a question back to the interviewer, and a filler
  phrase are not claims.
- The real reply is a speech transcript and can contain misheard words. When a
  word in the real reply is plainly a transcription error, read the real reply
  with the intended word.
- Mark a claim "contradicted" only when the real reply itself conflicts with
  the claim. A claim that is merely new is "absent".
- Use only the real reply as the reference. Do not use outside knowledge about
  the person to mark a claim "supported".
- When the generated reply makes no claims, return an empty `claims` list.
</rules>
"""


def judge_human_message(example: EvaluationExample, generated_answer: str) -> str:
    """Render the judge's human message for one example."""
    return (
        f"Question:\n{example.question}\n\n"
        f"Real reply:\n{example.real_answer}\n\n"
        f"Generated reply:\n{generated_answer}"
    )


@dataclass
class AccuracyScore:
    """How accurately one generated reply reproduces the real reply."""

    addresses_the_question: bool | None
    supported_claims: int
    contradicted_claims: int
    absent_claims: int
    missing_real_points: int
    # supported / (supported + contradicted + absent): of what the avatar
    # said, the share the person really said.
    claim_precision: float | None
    # supported / (supported + missing real points): of what the person
    # really said, the share the avatar reproduced.
    point_recall: float | None
    semantic_similarity: float | None


def accuracy_score_from_judgement(
    judgement: TranscriptAnswerJudgement | None,
    *,
    semantic_similarity: float | None,
) -> AccuracyScore:
    """Reduce a judgement to counts and the two ratios."""
    if judgement is None:
        return AccuracyScore(None, 0, 0, 0, 0, None, None, semantic_similarity)
    verdicts = [claim.verdict for claim in judgement.claims]
    supported = verdicts.count("supported")
    contradicted = verdicts.count("contradicted")
    absent = verdicts.count("absent")
    missing = len(judgement.real_reply_points_missing_from_generated_reply)
    total_claims = supported + contradicted + absent
    return AccuracyScore(
        addresses_the_question=judgement.addresses_the_question,
        supported_claims=supported,
        contradicted_claims=contradicted,
        absent_claims=absent,
        missing_real_points=missing,
        claim_precision=(supported / total_claims) if total_claims else None,
        point_recall=(supported / (supported + missing)) if (supported + missing) else None,
        semantic_similarity=semantic_similarity,
    )


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Cosine similarity of two embedding vectors, ``None`` for a zero vector."""
    dot_product = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(a * a for a in first))
    second_norm = math.sqrt(sum(b * b for b in second))
    if not first_norm or not second_norm:
        return None
    return dot_product / (first_norm * second_norm)


# ---------------------------------------------------------------------------
# Verisimilitude
# ---------------------------------------------------------------------------


@dataclass
class VerisimilitudeScore:
    """How much one generated reply reads like the person's own speech."""

    # Squared Mahalanobis distance of the reply's style features from the
    # avatar's direct-quote cloud, and the avatar's empirical threshold.
    distance_from_direct_quotes: float | None
    direct_quote_threshold: float | None
    # distance / threshold: below 1.0 the reply sits inside the cloud of the
    # person's own quotes.
    direct_quote_distance_ratio: float | None
    within_direct_quote_threshold: bool | None
    isolation_forest_accepts_reply: bool | None
    # Squared Mahalanobis distance from the unmodified-model baseline; at or
    # below the threshold the reply is not distinguishable from a stock model.
    distance_from_unmodified_llm_responses: float | None
    unmodified_llm_response_threshold: float | None
    indistinguishable_from_unmodified_llm: bool | None
    generated_word_count: int
    real_word_count: int


def verisimilitude_score_from_response_metadata(
    response_metadata: Dict[str, Any] | None,
    *,
    generated_answer: str,
    real_answer: str,
) -> VerisimilitudeScore:
    """Read the style analysis the message path attached to one reply."""
    metadata = response_metadata or {}
    distances = metadata.get("authenticity_distances") or {}
    direct_quote_analysis = metadata.get("comparison_to_direct_quote_response_analysis") or {}
    distance = _optional_float(distances.get("squared_mahalanobis_distance_from_direct_quotes"))
    threshold = _optional_float(distances.get("direct_quote_threshold"))
    baseline_distance = _optional_float(
        distances.get("squared_mahalanobis_distance_from_unmodified_llm_responses")
    )
    baseline_threshold = _optional_float(distances.get("unmodified_llm_response_threshold"))
    return VerisimilitudeScore(
        distance_from_direct_quotes=distance,
        direct_quote_threshold=threshold,
        direct_quote_distance_ratio=(
            distance / threshold if distance is not None and threshold else None
        ),
        within_direct_quote_threshold=(
            distance < threshold if distance is not None and threshold is not None else None
        ),
        isolation_forest_accepts_reply=direct_quote_analysis.get(
            "no_statisically_significant_difference_between_sample_and_direct_quotes_dataset_according_to_isolation_forest"
        ),
        distance_from_unmodified_llm_responses=baseline_distance,
        unmodified_llm_response_threshold=baseline_threshold,
        indistinguishable_from_unmodified_llm=(
            baseline_distance <= baseline_threshold
            if baseline_distance is not None and baseline_threshold is not None
            else None
        ),
        generated_word_count=word_count(generated_answer),
        real_word_count=word_count(real_answer),
    )


def compare_style_feature_means(
    generated_feature_rows: Sequence[Dict[str, float]],
    real_feature_rows: Sequence[Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Compare each style feature's mean over generated and over real replies.

    One reply is too short for most of the 28 style features to be stable, so
    the comparison is made over the whole run: the mean of each feature across
    every generated reply against the mean across every real reply to the same
    questions. ``standardized_difference`` is the difference of the two means
    divided by the standard deviation of the real replies' values, so 0.0
    means the avatar matches the person on that feature and 2.0 means the
    avatar sits two of the person's own standard deviations away. Features
    are returned largest absolute standardized difference first; a feature
    with fewer than two finite real values, or with no spread, has
    ``standardized_difference`` ``None`` and sorts last.
    """

    def _finite_values(rows: Sequence[Dict[str, float]], name: str) -> List[float]:
        values = []
        for row in rows:
            value = row.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        return values

    feature_names: List[str] = []
    for row in list(generated_feature_rows) + list(real_feature_rows):
        for name in row:
            if name not in feature_names:
                feature_names.append(name)

    comparison: List[Dict[str, Any]] = []
    for name in feature_names:
        generated_values = _finite_values(generated_feature_rows, name)
        real_values = _finite_values(real_feature_rows, name)
        generated_mean = sum(generated_values) / len(generated_values) if generated_values else None
        real_mean = sum(real_values) / len(real_values) if real_values else None
        standardized_difference = None
        if generated_mean is not None and len(real_values) >= 2:
            real_deviation = math.sqrt(
                sum((value - real_mean) ** 2 for value in real_values) / (len(real_values) - 1)
            )
            if real_deviation > 1e-12:
                standardized_difference = (generated_mean - real_mean) / real_deviation
        comparison.append(
            {
                "feature": name,
                "generated_mean": generated_mean,
                "real_mean": real_mean,
                "standardized_difference": standardized_difference,
                "generated_replies_measured": len(generated_values),
                "real_replies_measured": len(real_values),
            }
        )
    comparison.sort(
        key=lambda row: (
            row["standardized_difference"] is None,
            -abs(row["standardized_difference"] or 0.0),
        )
    )
    return comparison


# ---------------------------------------------------------------------------
# One example's local trace, and the run summary
# ---------------------------------------------------------------------------


@dataclass
class EvaluationTrace:
    """Everything recorded locally for one example: the trace of one question."""

    example: EvaluationExample
    generated_answer: str
    thread_id: str | None
    latency_milliseconds: float | None
    reply_usage: Dict[str, Any]
    judge_usage: Dict[str, Any]
    accuracy: AccuracyScore
    verisimilitude: VerisimilitudeScore
    judgement: Dict[str, Any] | None = None
    error: str | None = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dictionary(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _share(values: Iterable[bool | None]) -> float | None:
    known = [bool(value) for value in values if value is not None]
    return sum(known) / len(known) if known else None


def langsmith_trace_cost_projection(graph_invocations: int) -> Dict[str, Any]:
    """What sending this many avatar turns to LangSmith would cost.

    Derived from the one invoice that was measured (GEHMVWPY0003, 2026-10-03):
    $182.58 for 46,515 base traces, produced by 6,704 graph invocations.
    """
    dollars_per_trace = LANGSMITH_INVOICE_DOLLARS / LANGSMITH_INVOICE_BASE_TRACES
    traces_per_invocation = LANGSMITH_INVOICE_BASE_TRACES / LANGSMITH_INVOICE_GRAPH_INVOCATIONS
    projected_traces = graph_invocations * traces_per_invocation
    return {
        "graph_invocations": graph_invocations,
        "dollars_per_trace": dollars_per_trace,
        "traces_per_graph_invocation": traces_per_invocation,
        "projected_traces": projected_traces,
        "projected_dollars": projected_traces * dollars_per_trace,
        "arithmetic": (
            f"{graph_invocations} invocations x ({LANGSMITH_INVOICE_BASE_TRACES} traces / "
            f"{LANGSMITH_INVOICE_GRAPH_INVOCATIONS} invocations) x (${LANGSMITH_INVOICE_DOLLARS} / "
            f"{LANGSMITH_INVOICE_BASE_TRACES} traces) = ${projected_traces * dollars_per_trace:.5f}"
        ),
        "source": "LangSmith invoice GEHMVWPY0003 dated 2026-10-03",
    }


def summarize_evaluation(
    traces: Sequence[EvaluationTrace],
    *,
    style_feature_comparison: List[Dict[str, Any]] | None = None,
    reply_cost_dollars: float | None = None,
    judge_cost_dollars: float | None = None,
) -> Dict[str, Any]:
    """Aggregate a run's traces into the summary the report prints."""
    scored = [trace for trace in traces if trace.error is None]
    summary: Dict[str, Any] = {
        "examples": len(traces),
        "examples_answered": len(scored),
        "examples_failed": len(traces) - len(scored),
        "accuracy": {
            "share_of_replies_that_address_the_question": _share(
                trace.accuracy.addresses_the_question for trace in scored
            ),
            "mean_claim_precision": _mean(trace.accuracy.claim_precision for trace in scored),
            "mean_point_recall": _mean(trace.accuracy.point_recall for trace in scored),
            "mean_semantic_similarity": _mean(
                trace.accuracy.semantic_similarity for trace in scored
            ),
            "supported_claims": sum(trace.accuracy.supported_claims for trace in scored),
            "contradicted_claims": sum(trace.accuracy.contradicted_claims for trace in scored),
            "absent_claims": sum(trace.accuracy.absent_claims for trace in scored),
            "replies_with_a_contradicted_claim": sum(
                1 for trace in scored if trace.accuracy.contradicted_claims > 0
            ),
        },
        "verisimilitude": {
            "share_of_replies_within_direct_quote_threshold": _share(
                trace.verisimilitude.within_direct_quote_threshold for trace in scored
            ),
            "share_of_replies_accepted_by_isolation_forest": _share(
                trace.verisimilitude.isolation_forest_accepts_reply for trace in scored
            ),
            "share_of_replies_indistinguishable_from_unmodified_llm": _share(
                trace.verisimilitude.indistinguishable_from_unmodified_llm for trace in scored
            ),
            "mean_direct_quote_distance_ratio": _mean(
                trace.verisimilitude.direct_quote_distance_ratio for trace in scored
            ),
            "mean_generated_word_count": _mean(
                trace.verisimilitude.generated_word_count for trace in scored
            ),
            "mean_real_word_count": _mean(
                trace.verisimilitude.real_word_count for trace in scored
            ),
            "style_feature_comparison": style_feature_comparison or [],
        },
        "latency": {
            "mean_reply_latency_milliseconds": _mean(
                trace.latency_milliseconds for trace in scored
            ),
        },
        "cost": {
            "reply_cost_dollars": reply_cost_dollars,
            "judge_cost_dollars": judge_cost_dollars,
            "langsmith_traces_sent": 0,
            "langsmith_cost_if_these_turns_were_traced": langsmith_trace_cost_projection(
                len(traces)
            ),
        },
    }
    return summary


def _formatted(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_evaluation_report(
    summary: Dict[str, Any],
    traces: Sequence[EvaluationTrace],
    *,
    avatar_name: str,
    assistant_id: str,
    run_label: str,
) -> str:
    """Render the run as Markdown: summary, worst examples, and costs."""
    accuracy = summary["accuracy"]
    verisimilitude = summary["verisimilitude"]
    cost = summary["cost"]
    projection = cost["langsmith_cost_if_these_turns_were_traced"]
    lines = [
        f"# Transcript evaluation: {avatar_name}",
        "",
        f"- Avatar: `{assistant_id}`",
        f"- Run: {run_label}",
        f"- Examples: {summary['examples']} ({summary['examples_answered']} answered, "
        f"{summary['examples_failed']} failed)",
        "",
        "## Accuracy with respect to the conversation",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Replies that address the question | {_formatted(accuracy['share_of_replies_that_address_the_question'])} |",
        f"| Mean claim precision (supported claims / all generated claims) | {_formatted(accuracy['mean_claim_precision'])} |",
        f"| Mean point recall (supported claims / (supported + missing real points)) | {_formatted(accuracy['mean_point_recall'])} |",
        f"| Mean semantic similarity (embedding cosine) | {_formatted(accuracy['mean_semantic_similarity'])} |",
        f"| Claims supported / contradicted / absent | {accuracy['supported_claims']} / {accuracy['contradicted_claims']} / {accuracy['absent_claims']} |",
        f"| Replies with at least one contradicted claim | {accuracy['replies_with_a_contradicted_claim']} |",
        "",
        "## Verisimilitude of the generated text",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Replies within the direct-quote threshold | {_formatted(verisimilitude['share_of_replies_within_direct_quote_threshold'])} |",
        f"| Replies accepted by the isolation forest | {_formatted(verisimilitude['share_of_replies_accepted_by_isolation_forest'])} |",
        f"| Replies indistinguishable from an unmodified model | {_formatted(verisimilitude['share_of_replies_indistinguishable_from_unmodified_llm'])} |",
        f"| Mean distance / threshold against direct quotes (below 1.0 is inside the cloud) | {_formatted(verisimilitude['mean_direct_quote_distance_ratio'])} |",
        f"| Mean words per generated reply | {_formatted(verisimilitude['mean_generated_word_count'], 1)} |",
        f"| Mean words per real reply | {_formatted(verisimilitude['mean_real_word_count'], 1)} |",
        "",
    ]
    feature_rows = [
        row
        for row in verisimilitude["style_feature_comparison"]
        if row["standardized_difference"] is not None
    ][:10]
    if feature_rows:
        lines += [
            "### Style features furthest from the real replies",
            "",
            "Standardized difference = (generated mean - real mean) / standard deviation of the real replies.",
            "",
            "| Feature | Generated mean | Real mean | Standardized difference |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {row['feature']} | {_formatted(row['generated_mean'])} | "
            f"{_formatted(row['real_mean'])} | {_formatted(row['standardized_difference'], 2)} |"
            for row in feature_rows
        ]
        lines.append("")

    lines += [
        "## Cost",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Avatar replies | {_formatted(cost['reply_cost_dollars'], 5)} dollars |",
        f"| Judge calls | {_formatted(cost['judge_cost_dollars'], 5)} dollars |",
        f"| LangSmith traces sent by this run | {cost['langsmith_traces_sent']} |",
        f"| LangSmith cost if these turns were traced | {projection['projected_dollars']:.5f} dollars |",
        "",
        f"LangSmith projection: {projection['arithmetic']} (source: {projection['source']}).",
        "",
        "## Examples, lowest claim precision first",
        "",
    ]
    ordered = sorted(
        (trace for trace in traces if trace.error is None),
        key=lambda trace: (
            trace.accuracy.claim_precision is None,
            trace.accuracy.claim_precision or 0.0,
        ),
    )
    for trace in ordered[:15]:
        lines += [
            f"### {trace.example.example_id}",
            "",
            f"- Claim precision: {_formatted(trace.accuracy.claim_precision)}; "
            f"point recall: {_formatted(trace.accuracy.point_recall)}; "
            f"semantic similarity: {_formatted(trace.accuracy.semantic_similarity)}; "
            f"direct-quote distance ratio: {_formatted(trace.verisimilitude.direct_quote_distance_ratio)}",
            f"- Question: {trace.example.question}",
            f"- Real reply: {trace.example.real_answer}",
            f"- Generated reply: {trace.generated_answer}",
            "",
        ]
    failed = [trace for trace in traces if trace.error is not None]
    if failed:
        lines += ["## Failed examples", ""]
        lines += [f"- {trace.example.example_id}: {trace.error}" for trace in failed]
        lines.append("")
    return "\n".join(lines)
