"""Scoring an avatar's replies against the real person's replies in a transcript.

- Examples come from the stored question and answer dataset, and an example
  counts as a real question only when a stored quote row records that same
  question as what was really said before the answer.
- Selection keeps real questions with real answers and spreads a limit evenly
  across the source recordings.
- Accuracy reduces the judge's claim verdicts to claim precision and point
  recall; verisimilitude reads the style distances the message path attached.
- The LangSmith projection is the measured invoice arithmetic, and a run
  reports zero traces sent.
"""

import json

import pytest

from src.anubis.utils.evaluation.transcript_evaluation import (
    EvaluationExample,
    EvaluationTrace,
    JudgedClaim,
    TranscriptAnswerJudgement,
    accuracy_score_from_judgement,
    build_evaluation_examples,
    compare_style_feature_means,
    cosine_similarity,
    langsmith_trace_cost_projection,
    render_evaluation_report,
    select_evaluation_examples,
    summarize_evaluation,
    verisimilitude_score_from_response_metadata,
)

SOURCE = "https://www.youtube.com/watch?v=VIDEO123456"
REAL_ANSWER = "I think the most important thing is to work hard every single day on something you love."


def _dataset_row(examples, source=SOURCE):
    return {"value": json.dumps(examples), "source_filename": source, "row_count": len(examples)}


def _stored_example(question, answer, source=SOURCE):
    return {
        "inputs": {"question": question},
        "outputs": {"answer": answer},
        "metadata": {"source": source},
    }


def _quote_row(answer, adapter_prompt, source=SOURCE, start=12.0, end=20.0):
    return {
        "page_content": answer,
        "metadata": {
            "source": source,
            "adapter_prompt": adapter_prompt,
            "start": start,
            "end": end,
        },
    }


def test_a_question_is_real_only_when_a_quote_row_records_the_same_question():
    examples = build_evaluation_examples(
        [
            _dataset_row(
                [
                    _stored_example("What matters most to you in life?", REAL_ANSWER),
                    _stored_example("A question a model wrote afterwards?", "Another real answer."),
                ]
            )
        ],
        [
            _quote_row(REAL_ANSWER, "What matters most to you  in life?"),
            _quote_row("Another real answer.", None),
        ],
    )
    assert [example.question_is_real for example in examples] == [True, False]
    assert examples[0].start_seconds == 12.0
    assert examples[0].example_id == f"{SOURCE}#0"


def test_selection_keeps_real_questions_with_real_answers():
    examples = [
        EvaluationExample("a", SOURCE, "What matters most to you in life?", REAL_ANSWER, True),
        EvaluationExample("b", SOURCE, "Really?", REAL_ANSWER, True),
        EvaluationExample("c", SOURCE, "What matters most to you in life?", "Yeah.", True),
        EvaluationExample("d", SOURCE, "What matters most to you in life?", REAL_ANSWER, False),
    ]
    assert [example.example_id for example in select_evaluation_examples(examples)] == ["a"]
    assert [
        example.example_id
        for example in select_evaluation_examples(examples, real_questions_only=False)
    ] == ["a", "d"]


def test_a_limit_is_spread_across_the_source_recordings():
    examples = [
        EvaluationExample(f"long-{index}", "long", "What do you think about that?", REAL_ANSWER, True)
        for index in range(10)
    ] + [EvaluationExample("short-0", "short", "What do you think about that?", REAL_ANSWER, True)]
    selected = select_evaluation_examples(examples, limit=4)
    assert [example.example_id for example in selected] == [
        "long-0",
        "short-0",
        "long-1",
        "long-2",
    ]


def test_claim_precision_and_point_recall():
    judgement = TranscriptAnswerJudgement(
        addresses_the_question=True,
        claims=[
            JudgedClaim(claim="Work hard every day.", verdict="supported"),
            JudgedClaim(claim="Love what you do.", verdict="supported"),
            JudgedClaim(claim="I grew up in Texas.", verdict="absent"),
            JudgedClaim(claim="Hard work does not matter.", verdict="contradicted"),
        ],
        real_reply_points_missing_from_generated_reply=["Do the work every single day."],
        explanation="Two of four claims match.",
    )
    score = accuracy_score_from_judgement(judgement, semantic_similarity=0.5)
    # 2 supported / 4 generated claims
    assert score.claim_precision == pytest.approx(0.5)
    # 2 supported / (2 supported + 1 missing real point)
    assert score.point_recall == pytest.approx(2 / 3)
    assert (score.supported_claims, score.contradicted_claims, score.absent_claims) == (2, 1, 1)


def test_a_reply_with_no_claims_has_no_precision_rather_than_zero():
    judgement = TranscriptAnswerJudgement(
        addresses_the_question=False,
        claims=[],
        real_reply_points_missing_from_generated_reply=["Work hard."],
        explanation="The reply deflects.",
    )
    score = accuracy_score_from_judgement(judgement, semantic_similarity=None)
    assert score.claim_precision is None
    assert score.point_recall == 0.0


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None


def test_verisimilitude_reads_the_distances_the_message_path_attached():
    score = verisimilitude_score_from_response_metadata(
        {
            "authenticity_distances": {
                "squared_mahalanobis_distance_from_direct_quotes": 30.0,
                "direct_quote_threshold": 40.0,
                "squared_mahalanobis_distance_from_unmodified_llm_responses": 90.0,
                "unmodified_llm_response_threshold": 50.0,
            },
            "comparison_to_direct_quote_response_analysis": {
                "no_statisically_significant_difference_between_sample_and_direct_quotes_dataset_according_to_isolation_forest": True
            },
        },
        generated_answer="one two three",
        real_answer="one two three four five",
    )
    # 30.0 / 40.0
    assert score.direct_quote_distance_ratio == pytest.approx(0.75)
    assert score.within_direct_quote_threshold is True
    assert score.isolation_forest_accepts_reply is True
    assert score.indistinguishable_from_unmodified_llm is False
    assert (score.generated_word_count, score.real_word_count) == (3, 5)


def test_missing_style_analysis_is_not_measured_rather_than_a_pass():
    score = verisimilitude_score_from_response_metadata(
        {}, generated_answer="one", real_answer="two"
    )
    assert score.direct_quote_distance_ratio is None
    assert score.within_direct_quote_threshold is None


def test_style_features_are_compared_in_the_real_replies_own_standard_deviations():
    comparison = compare_style_feature_means(
        [{"mean_sentence_length": 30.0, "comma_rate": 0.1}] * 3,
        [
            {"mean_sentence_length": 8.0, "comma_rate": 0.1},
            {"mean_sentence_length": 10.0, "comma_rate": 0.1},
            {"mean_sentence_length": 12.0, "comma_rate": 0.1},
        ],
    )
    assert comparison[0]["feature"] == "mean_sentence_length"
    # (30.0 - 10.0) / sample standard deviation 2.0
    assert comparison[0]["standardized_difference"] == pytest.approx(10.0)
    # No spread among the real replies: the difference is not measured.
    assert comparison[1]["standardized_difference"] is None


def test_the_langsmith_projection_is_the_measured_invoice_arithmetic():
    projection = langsmith_trace_cost_projection(6704)
    # 6,704 invocations is exactly the invoiced window, so the projection is the invoice.
    assert projection["projected_traces"] == pytest.approx(46515)
    assert projection["projected_dollars"] == pytest.approx(182.58)
    assert langsmith_trace_cost_projection(0)["projected_dollars"] == 0.0


def _trace(example_id, *, precision_claims=("supported",), error=None):
    example = EvaluationExample(example_id, SOURCE, "What matters most to you?", REAL_ANSWER, True)
    judgement = TranscriptAnswerJudgement(
        addresses_the_question=True,
        claims=[JudgedClaim(claim="x", verdict=verdict) for verdict in precision_claims],
        real_reply_points_missing_from_generated_reply=[],
        explanation="",
    )
    return EvaluationTrace(
        example=example,
        generated_answer="" if error else "Work hard every day.",
        thread_id=None if error else f"thread-{example_id}",
        latency_milliseconds=None if error else 1000.0,
        reply_usage={},
        judge_usage={},
        accuracy=accuracy_score_from_judgement(
            None if error else judgement, semantic_similarity=None if error else 0.8
        ),
        verisimilitude=verisimilitude_score_from_response_metadata(
            None, generated_answer="Work hard every day.", real_answer=REAL_ANSWER
        ),
        error=error,
    )


def test_the_summary_counts_failures_separately_and_sends_no_traces():
    traces = [
        _trace("a", precision_claims=("supported", "supported")),
        _trace("b", precision_claims=("supported", "contradicted")),
        _trace("c", error="HTTPStatusError: 500"),
    ]
    summary = summarize_evaluation(traces, reply_cost_dollars=0.01, judge_cost_dollars=0.002)

    assert (summary["examples"], summary["examples_answered"], summary["examples_failed"]) == (3, 2, 1)
    # (1.0 + 0.5) / 2 answered examples
    assert summary["accuracy"]["mean_claim_precision"] == pytest.approx(0.75)
    assert summary["accuracy"]["replies_with_a_contradicted_claim"] == 1
    assert summary["cost"]["langsmith_traces_sent"] == 0
    assert summary["cost"]["langsmith_cost_if_these_turns_were_traced"]["graph_invocations"] == 3

    report = render_evaluation_report(
        summary, traces, avatar_name="Guest Person", assistant_id="assistant", run_label="run"
    )
    assert "## Accuracy with respect to the conversation" in report
    assert "## Verisimilitude of the generated text" in report
    assert "HTTPStatusError: 500" in report
    # A measure with no data is reported as not measured, never as zero.
    assert "not measured" in report
