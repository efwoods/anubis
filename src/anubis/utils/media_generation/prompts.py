"""The emotion set and the generation prompts, in one place.

The seven base emotions are exactly the values ``EMOTION_MAPPING``
(``src/anubis/utils/emotion_mapping.py``) collapses the twenty-eight GoEmotions
labels into, so every classified reply names an emotion that has a still and a
loop. ``neutral`` is the reference image itself; the other six are generated
from it.

Prompts are written against the xAI image-editing and image-to-video models.
The still prompts change ONLY the expression: same person, framing, lighting,
clothing, background. The idle-loop prompt asks for breathing, blinks, and small
weight shifts — no speech, no mouth movement, no camera movement — and states
that the first and the final frame must match the supplied image, because the
loop is played end to end and any drift shows as a seam.
"""

from __future__ import annotations

NEUTRAL_EMOTION = "neutral"

GENERATED_EMOTIONS: tuple[str, ...] = (
    "joy",
    "anger",
    "sadness",
    "fear",
    "surprise",
    "disgust",
)

BASE_EMOTIONS: tuple[str, ...] = (NEUTRAL_EMOTION, *GENERATED_EMOTIONS)

_EXPRESSION_BY_EMOTION: dict[str, str] = {
    "joy": (
        "clear, unmistakable joy: a genuine smile with raised cheeks, crinkled "
        "eyes, and an open, warm face"
    ),
    "anger": (
        "clear, unmistakable anger: lowered and drawn-together brows, a hard "
        "stare, tightened lips, and tension in the jaw"
    ),
    "sadness": (
        "clear, unmistakable sadness: inner brows raised and drawn together, "
        "downcast eyes, and the corners of the mouth turned down"
    ),
    "fear": (
        "clear, unmistakable fear: raised and drawn-together brows, widened "
        "eyes showing more white, and lips stretched horizontally"
    ),
    "surprise": (
        "clear, unmistakable surprise: raised, arched brows, widened eyes, and "
        "a slightly dropped, open jaw"
    ),
    "disgust": (
        "clear, unmistakable disgust: a wrinkled nose, a raised upper lip, "
        "narrowed eyes, and the head drawn slightly back"
    ),
}

STILL_PROMPT_TEMPLATE = (
    "Same person, same framing, same lighting, same clothing, same background, "
    "same camera angle. Change only the facial expression to {expression}. Do "
    "not change the pose, the hair, the clothing, the camera angle, or the "
    "composition. Photorealistic, matching the source image exactly in every "
    "respect other than the expression."
)

IDLE_LOOP_PROMPT_TEMPLATE = (
    "Base idle animation of the person in the image, holding a {emotion} "
    "expression throughout. The person breathes naturally, blinks, and shifts "
    "weight slightly, with an occasional subtle fidget. No speech and no mouth "
    "movement other than breathing. No camera movement, no zoom, no background "
    "change, no new objects. The very first frame and the very last frame MUST "
    "match the supplied image exactly — the same pose, framing, and expression — "
    "so the clip loops seamlessly when played end to end."
)


def still_prompt_for(emotion: str) -> str:
    """Return the image-edit prompt that turns the reference into ``emotion``."""
    expression = _EXPRESSION_BY_EMOTION.get(emotion)
    if expression is None:
        raise ValueError(f"No still prompt is defined for emotion {emotion!r}.")
    return STILL_PROMPT_TEMPLATE.format(expression=expression)


def idle_loop_prompt_for(emotion: str) -> str:
    """Return the image-to-video prompt for ``emotion``'s idle loop."""
    label = "neutral, relaxed" if emotion == NEUTRAL_EMOTION else emotion
    return IDLE_LOOP_PROMPT_TEMPLATE.format(emotion=label)
