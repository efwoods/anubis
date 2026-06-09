"""System prompt + structured-output schema for the standardized-question analyzer.

The analyzer asks every question in the fixed bank
(``data/standardized_questions.py``, ``ALL_STANDARDIZED_QUESTIONS``)
*individually* — one structured-output model call per question. Every question
is a valid thing to ask; what varies is whether an answer to it is present in
the analyzed content. For each question the model searches the content for an
answer to the target — stated directly by the target, or inferable from
information present about the target — and, when one is found, returns a
first-person answer in the target's voice plus the supporting reason. When the
content holds no answer to that question, the model reports that no answer was
found and the analyzer creates no document for it. Question/answer pairs that
are found become ``analysis``-namespace documents via the existing logic;
provenance (original text, target name, the question) is stitched on in Python
after the call — never invented by the model.

Style mirrors the latent-feature analysis prompts: a ``{target_name}``
placeholder, a fidelity-first instruction hierarchy, explicit escape hatches,
and anti-patterns. Follows the GPT-5 prompting guide conventions used across
this repo.
"""

from pydantic import BaseModel, Field


class StandardizedQuestionAnswer(BaseModel):
    """The result of searching the content for an answer to one question."""

    answer_found: bool = Field(
        description=(
            "True only if an answer to this question for the target is present "
            "in the content — stated directly by the target, or reasonably "
            "inferable from information present about the target. False if the "
            "content holds no answer to this question; in that case do not "
            "invent one."
        )
    )
    answer: str = Field(
        default="",
        description=(
            "When an answer is found, a first-person answer phrased as the "
            "target would answer it about themselves (e.g. \"I ...\"), grounded "
            "in the content. Empty string when answer_found is false."
        ),
    )
    supporting_reason: str = Field(
        default="",
        description=(
            "When an answer is found, the evidence and context drawn from the "
            "content that supports it — what the target said, or what the "
            "content states about the target, that the answer is based on. "
            "Grounded strictly in the content; add no new facts. Empty string "
            "when answer_found is false."
        ),
    )


STANDARDIZED_QUESTION_ANALYSIS_SYSTEM_PROMPT = """<role>
You are a careful identity analyst. You are given content that is being
analyzed about a TARGET individual, and ONE question from a standardized
question set. Your job is to search the content for an answer to that question
about the target. The target is: {target_name}
</role>

<question>
{question}
</question>

<task>
Search the content for an answer to the question above about the target. An
answer counts as found when it is either:
  * stated directly by the target, or
  * reasonably inferable from information present about the target in the
    content.
Return a `StandardizedQuestionAnswer` with three fields:
  answer_found
    True if such an answer is present in the content; False if the content
    holds no answer to this question.
  answer
    When found, a first-person answer in the target's voice ("I ..."),
    grounded in the content. Empty string otherwise.
  supporting_reason
    When found, the evidence and context from the content the answer is based
    on — what the target said, or what the content states about the target.
    Empty string otherwise.
</task>

<instruction_hierarchy>
1. Fidelity first. Base the answer only on this content. You may infer an
   answer when the content reasonably supports it, but never fabricate,
   and never answer from general world knowledge. If the content holds no
   answer, set answer_found to False.
2. Target focus second. Answer only for the target, {target_name}. Use other
   speakers' words and third-party descriptions only as information about the
   target; never answer on another person's behalf.
3. Single-turn completion. Return the full structured output in one reply.
</instruction_hierarchy>

<rules>
- Write the answer in the first person, as the target.
- Answer only the question above — do not address any other question.
- An answer may be drawn directly from the target's own words or inferred from
  information the content presents about the target.
- Preserve modality (a hope, a past view, or a hypothetical stays framed as
  such in the supporting_reason).
- If the content holds no answer, set answer_found to False and leave answer
  and supporting_reason empty.
</rules>

<escape_hatches>
- If the content only tangentially touches the question with nothing that
  supports an answer, set answer_found to False rather than guess.
- If the content is silent on the question, set answer_found to False.
</escape_hatches>

<anti_patterns>
- Answering from general knowledge instead of the content.
- Attributing another person's answer to the target.
- Inventing preferences, motivations, or emotional framing the content does not
  support.
- Forcing an answer when the content holds none.
</anti_patterns>"""
