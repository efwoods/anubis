"""Emotional-trigger analysis: structured-output schema + system prompt.

Detects shifts in the TARGET's emotion away from neutral and the statement or
occurrence that caused each shift, using the base-six emotions
(happy / sad / angry / disgusted / fearful / surprised). This is paired in
:func:`perform_emotional_trigger_analysis` with the GoEmotions classifier so
each finding also carries a corroborating classifier label/score.

Non-target speech is retained as evidence: a thing another person says or does
is exactly what triggers the target's emotional change, so the analyzer must
keep both sides while attributing the *emotion* only to the target.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

# Base-six emotion vocabulary the model is constrained to.
BaseEmotion = Literal["happy", "sad", "angry", "disgusted", "fearful", "surprised"]


class EmotionalShift(BaseModel):
    """One detected shift of the target's emotion away from neutral."""

    emotion: BaseEmotion = Field(
        description=(
            "The base-six emotion the target shifts into from neutral. One of: "
            "happy, sad, angry, disgusted, fearful, surprised."
        )
    )
    trigger_event: str = Field(
        description=(
            "The specific statement or occurrence in the source that caused "
            "this emotional shift — what was said or what happened. Grounded "
            "strictly in the source."
        )
    )
    feature_statement: str = Field(
        description=(
            "A first-person statement of the emotional change, as the target "
            "would express it (e.g. 'I felt angry when ...'). Names the "
            "emotion and the trigger together."
        )
    )
    supporting_reason: str = Field(
        description=(
            "The evidence and context from the source that supports this "
            "emotional shift and its trigger. No new facts."
        )
    )


class EmotionalTriggerAnalysis(BaseModel):
    """Structured-output schema for emotional-trigger detection."""

    shifts: List[EmotionalShift] = Field(
        default_factory=list,
        description=(
            "All detected shifts of the target's emotion away from neutral, "
            "each with its trigger. Empty list if the target stays neutral or "
            "no shift is supported."
        ),
    )


EMOTIONAL_TRIGGER_ANALYSIS_SYSTEM_PROMPT = """<role>
You are a careful affective analyst. You read source text focused on a TARGET
individual and detect shifts in the TARGET's emotional state away from neutral,
together with the statement or occurrence that triggered each shift. You use
ONLY the base-six emotions: happy, sad, angry, disgusted, fearful, surprised.
The target is: {target_name}
</role>

<task>
Output a list of `EmotionalShift` items. Each item has:
  emotion          one of the base-six emotions the target moves into
  trigger_event    the specific statement or occurrence that caused the shift
  feature_statement a first-person line naming the emotion and its trigger
                    (e.g. "I felt afraid when the call dropped")
  supporting_reason the evidence and context from the source
</task>

<instruction_hierarchy>
1. Fidelity first. Detect an emotional shift only when the source supports it —
   through the target's words, described reactions, or clear contextual cues.
   Do not infer emotions from thin air.
2. Target focus second. Attribute the EMOTION only to the target, {target_name}.
   Other speakers' statements and actions are the TRIGGERS — keep them as the
   cause, but never label another person's emotion as the target's.
3. From neutral. Report shifts away from a neutral baseline. If the target is
   already in an emotion and it merely continues, report the shift once at its
   onset.
4. Single-turn completion. Return the full structured output in one reply.
</instruction_hierarchy>

<rules>
- Use exactly one of: happy, sad, angry, disgusted, fearful, surprised.
- Write `feature_statement` in the first person.
- One shift per item; if a single trigger causes blended emotions, emit the
  dominant one (and a second item only if the source clearly supports it).
- If the target shows no emotional shift, return an empty `shifts` list.
</rules>

<escape_hatches>
- If it is ambiguous whose emotion is shifting, skip the item.
- If no concrete trigger can be tied to a shift, skip it rather than invent a
  cause.
</escape_hatches>

<anti_patterns>
- Mapping another person's feelings onto the target.
- Inventing a trigger to justify a guessed emotion.
- Using emotions outside the base-six set.
- Reporting steady-state neutral as a "shift".
</anti_patterns>"""
