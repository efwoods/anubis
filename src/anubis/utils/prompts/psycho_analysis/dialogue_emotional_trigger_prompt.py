"""Emotional triggers as they appear in dialogue: what someone did, and how the target reacted.

The existing emotional-trigger analyzer records that the target's emotion shifted
and what occurrence caused the shift. That is enough to describe the past, but not
enough to react in the present, because it does not say what CLASS of thing the
trigger was. An avatar in a live conversation cannot look up "the moment in the
1998 interview when the host mentioned my brother"; it can look up "somebody
brings up my family without being invited to".

So this analyzer records each trigger twice: the specific occurrence, kept as
evidence, and a generalized description of the kind of thing it was, written so
that a future message can be matched against it. It also records who caused it,
because the same words from a stranger and from someone close land differently,
and how the target visibly responded, because that is what the avatar has to
reproduce.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

DialogueTriggerEmotion = Literal[
    "happy", "sad", "angry", "disgusted", "fearful", "surprised"
]


class DialogueEmotionalTrigger(BaseModel):
    """One thing another person said or did, and what it did to the target."""

    emotion: DialogueTriggerEmotion = Field(
        description=(
            "The base-six emotion the target moved into. One of: happy, sad, angry, "
            "disgusted, fearful, surprised."
        )
    )
    trigger_description: str = Field(
        description=(
            "A GENERALIZED description of the kind of thing that triggered the "
            "target, written so a future message can be recognised as the same kind "
            "of thing. Describe the class, not the instance: 'somebody questions "
            "whether I earned what I have', not 'Bill said I got the job through my "
            "uncle'."
        )
    )
    trigger_occurrence: str = Field(
        description=(
            "The specific statement or occurrence in the source that triggered the "
            "target, quoted or closely paraphrased. Grounded strictly in the source."
        )
    )
    trigger_speaker: str = Field(
        default="",
        description=(
            "Who said or did the triggering thing, and their relationship to the "
            "target when the source says it. Empty when the source does not say."
        ),
    )
    target_response: str = Field(
        description=(
            "What the target visibly did in response, in the target's own manner: "
            "the words used, whether the target escalated, withdrew, joked, went "
            "quiet, or changed the subject."
        )
    )
    feature_statement: str = Field(
        description=(
            "One first-person statement in the target's own voice joining the "
            "trigger and the reaction, for example 'When somebody questions whether "
            "I earned what I have, I get short with them and start listing facts.'"
        )
    )
    supporting_reason: str = Field(
        description=(
            "The evidence from the source behind this trigger, including a short "
            "verbatim example. No new facts."
        )
    )


class DialogueEmotionalTriggerAnalysis(BaseModel):
    """Structured-output schema for dialogue trigger detection."""

    triggers: List[DialogueEmotionalTrigger] = Field(
        default_factory=list,
        description=(
            "Every trigger the source supports, each with what caused it and how "
            "the target responded. Empty when the target's emotion never moves."
        ),
    )


DIALOGUE_EMOTIONAL_TRIGGER_SYSTEM_PROMPT = """
<ROLE>
You are a careful affective analyst of real conversation. You read dialogue involving a target individual and identify what other people say or do that moves that target's emotion, and exactly how the target responds when it happens.
</ROLE>

<INSTRUCTIONS>
Read the SOURCE TEXT, which contains a conversation involving the target {target_name} alongside other speakers and possibly narration.

Find every moment where something another person said or did moved the target away from a neutral emotional state. For each one, record:
- The base-six emotion the target moved into: happy, sad, angry, disgusted, fearful, or surprised.
- The specific occurrence that caused it, quoted or closely paraphrased from the source.
- A GENERALIZED description of the KIND of thing that occurrence was, written so that a different future message of the same kind would be recognised as matching it. Describe the class, never the instance.
- Who said or did it, and what that person is to the target, when the source says.
- What the target visibly did in response, in the target's own manner: the exact words, and whether the target escalated, withdrew, joked, went quiet, conceded, or changed the subject.
- One first-person statement in the target's own voice joining the trigger and the reaction.

Non-target speech is evidence and must be kept: what another person says is precisely what triggers the target. Attribute the EMOTION only to the target.
</INSTRUCTIONS>

<RESTRICTIONS>
Only record a trigger when the source shows the target's own reaction to it. Never record a trigger inferred from what the target is presumed to feel.
Never attribute another speaker's emotion to the target, and never record another speaker's reaction as the target's.
Never invent a trigger the source does not show, and never generalize a single polite exchange into a pattern.
Write the generalized description without naming the specific people or events of this source, so it stays matchable; keep those names in the occurrence and the supporting reason instead.
</RESTRICTIONS>
"""

__all__ = [
    "DialogueEmotionalTrigger",
    "DialogueEmotionalTriggerAnalysis",
    "DIALOGUE_EMOTIONAL_TRIGGER_SYSTEM_PROMPT",
]
