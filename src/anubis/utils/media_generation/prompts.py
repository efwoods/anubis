"""The emotion set and the generation prompts, in one place.

The seven base emotions are exactly the values ``EMOTION_MAPPING``
(``src/anubis/utils/emotion_mapping.py``) collapses the twenty-eight GoEmotions
labels into, so every classified reply names an emotion that has a still and a
loop. ``neutral`` is the reference image itself; the other six are generated
from it.

Prompts are written against the xAI image-editing and image-to-video models,
and every prompt is chosen by the **reference subject**: a reference image is a
photograph of a person, a stylized character (an illustration, a game render, a
cartoon), or something with no face at all (a logo, an abstract interface, an
object, a landscape). The person and character families change ONLY the facial
expression: same subject, framing, lighting, clothing, background. The
non-human family never mentions a face or a person — an editing model told to
"change the facial expression" of an image with no face invents a person to
carry the expression, which is how an abstract heads-up display turned into a
crowd of strangers — and instead expresses each emotion through color
temperature, brightness, and the motion of the shapes already in the image.

The idle-loop prompts ask for breathing, blinks, and small weight shifts (for a
face) or a slow ambient pulse and drift (for a non-human subject) — no speech,
no mouth movement, no camera movement — and state that the first and the final
frame must match the supplied image, because the loop is played end to end and
any drift shows as a seam.
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

# What the reference image depicts. Decides which prompt family generates the
# emotion media. ``person`` is the default and the behaviour that existed
# before subjects were classified.
SUBJECT_PERSON = "person"
SUBJECT_STYLIZED_CHARACTER = "stylized_character"
SUBJECT_NON_HUMAN = "non_human"

REFERENCE_SUBJECTS: tuple[str, ...] = (
    SUBJECT_PERSON,
    SUBJECT_STYLIZED_CHARACTER,
    SUBJECT_NON_HUMAN,
)

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

# How a subject with no face shows each emotion: light, glow, contrast, and
# the shape of its own elements. Nothing here names a body part, and nothing
# replaces the subject's identifying color — a red lens stays a red lens and
# only its intensity, spread, and a slight hue shift carry the feeling.
_NON_HUMAN_CUE_BY_EMOTION: dict[str, str] = {
    "joy": (
        "joy: warm, bright, and open — the glow brighter and wider, the "
        "highlights lifted, the whole image lighter, the elements gently "
        "expanded and lifted, the hue warmed only slightly toward gold"
    ),
    "anger": (
        "anger: hot, hard, and tight — the glow intense and saturated with "
        "hard contrast, the brightest core burning sharper, the edges "
        "sharpened and tightened, the hue pushed only slightly hotter"
    ),
    "sadness": (
        "sadness: dim, cool, and heavy — the brightness lowered, the glow "
        "shrunk and softened, the contrast reduced, the elements settled "
        "lower, the hue cooled and desaturated only slightly"
    ),
    "fear": (
        "fear: pale, unsteady, and contracted — the glow thinned and "
        "uneven as if flickering, the brightest core shrunk, thin pale "
        "streaks of light, the elements drawn inward"
    ),
    "surprise": (
        "surprise: a sudden flare — the glow flung wide and bright, the "
        "highlights nearly blown out, the elements widened outward as if "
        "startled"
    ),
    "disgust": (
        "disgust: sour and recoiling — a faint sickly tint over the subject's "
        "own color, the glow pulled back and uneven, the edges warped and "
        "curdled, the elements drawn slightly away"
    ),
}

_NON_HUMAN_LOOP_MOTION_BY_EMOTION: dict[str, str] = {
    NEUTRAL_EMOTION: "a calm, even pulse and a slow, steady drift",
    "joy": "a bright, buoyant pulse and a gentle upward drift",
    "anger": "a hard, fast flicker and sharp, jagged pulses",
    "sadness": "a slow, dim pulse and a heavy downward drift",
    "fear": "a tremulous, unsteady flicker and a tight inward contraction",
    "surprise": "a sudden outward flare that settles back",
    "disgust": "a recoiling contraction and a queasy, warped wobble",
}

_PERSON_STILL_PROMPT_TEMPLATE = (
    "Same person, same framing, same lighting, same clothing, same background, "
    "same camera angle. Change only the facial expression to {expression}. Do "
    "not change the pose, the hair, the clothing, the camera angle, or the "
    "composition. Photorealistic, matching the source image exactly in every "
    "respect other than the expression.{motion}"
)

# When the person's own movement has been measured, the still already holds
# that person's characteristic carriage — head tilt, lean, where the hands
# rest — so anything animating the still inherits it. Only the carriage is
# taken from the block; the still is a single frame and cannot show a rate.
_PERSON_STILL_MOTION_TEMPLATE = (
    " The person's posture and carriage in this still match how this person "
    "really holds themself, measured from this person: {motion_prompt}"
)

_CHARACTER_STILL_PROMPT_TEMPLATE = (
    "Same character, same art style, same framing, same lighting, same "
    "clothing, same background, same camera angle. Change only the facial "
    "expression to {expression}. Do not change the pose, the hair, the "
    "clothing, the camera angle, or the composition. Keep the exact rendering "
    "style of the source image — do not make the image photorealistic and do "
    "not redraw the character — matching the source image exactly in every "
    "respect other than the expression.{motion}"
)

_NON_HUMAN_STILL_PROMPT_TEMPLATE = (
    "Same subject, same framing, same composition, same camera angle, same "
    "background. This image contains no person and no face: do not add a "
    "person, a face, a figure, eyes, a mouth, or any body part, and do not "
    "turn the subject into a character. Express {cue}. Work only with the "
    "light, the glow, the brightness, the contrast, and the shape and spacing "
    "of the elements already present. Keep the subject's own identifying "
    "colors and materials — shift their hue only slightly toward the emotion, "
    "never replace them. Keep the exact style of the source image, matching "
    "it in every respect other than that emotional lighting."
)

# The generic idle motion, used only until the person's own movement has been
# measured. Once a motion block exists, it replaces this clause entirely: the
# clip breathes, blinks, shifts and fidgets the way *this* person does.
_GENERIC_PERSON_IDLE_MOTION = (
    "The person breathes naturally, blinks, and shifts weight slightly, with an "
    "occasional subtle fidget."
)
_MEASURED_PERSON_IDLE_MOTION_TEMPLATE = (
    "The person breathes naturally and moves the way this person really moves "
    "when idle, described from measurements of this person — keep every rate, "
    "range and habit below, and add nothing else:\n{motion_prompt}"
)

_PERSON_IDLE_LOOP_PROMPT_TEMPLATE = (
    "Base idle animation of the person in the image, holding a {emotion} "
    "expression throughout. {motion} No speech and no mouth "
    "movement other than breathing. No camera movement, no zoom, no background "
    "change, no new objects. The very first frame and the very last frame MUST "
    "match the supplied image exactly — the same pose, framing, and expression — "
    "so the clip loops seamlessly when played end to end."
)

_GENERIC_CHARACTER_IDLE_MOTION = (
    "The character breathes naturally, blinks, and shifts weight slightly, with "
    "an occasional subtle fidget."
)
_MEASURED_CHARACTER_IDLE_MOTION_TEMPLATE = (
    "The character breathes naturally and moves the way the real person behind "
    "this character moves when idle, described from measurements of that person "
    "— keep every rate, range and habit below, and add nothing else:\n{motion_prompt}"
)

_CHARACTER_IDLE_LOOP_PROMPT_TEMPLATE = (
    "Base idle animation of the character in the image, in the exact art "
    "style of the image, holding a {emotion} expression throughout. {motion} "
    "No speech and no mouth movement other than "
    "breathing. No camera movement, no zoom, no background change, no new "
    "objects, no change of rendering style. The very first frame and the very "
    "last frame MUST match the supplied image exactly — the same pose, framing, "
    "and expression — so the clip loops seamlessly when played end to end."
)

_NON_HUMAN_IDLE_LOOP_PROMPT_TEMPLATE = (
    "Ambient idle animation of the subject in the image, holding its {emotion} "
    "lighting throughout: {motion}. The light itself pulses — its glow swells "
    "and settles slowly — and nothing else changes. This image contains no person "
    "and no face: do not add a person, a face, a figure, or any body part, and "
    "do not turn the subject into a character. Only the elements already "
    "present move, and the subject keeps its own colors. No camera movement, "
    "no zoom, no background change, no new objects. The very first frame and "
    "the very last frame MUST match the supplied image exactly — the same "
    "composition, framing, and lighting — so the clip loops seamlessly when "
    "played end to end."
)


def normalize_reference_subject(subject: str | None) -> str:
    """Map any stored or supplied subject onto a known family, defaulting to person."""
    candidate = str(subject or "").strip().lower()
    return candidate if candidate in REFERENCE_SUBJECTS else SUBJECT_PERSON


def _posture_only(motion_prompt: str | None) -> str:
    """Keep the lines of a motion block a single frame can show (posture, hands)."""
    lines = [
        line.strip()
        for line in str(motion_prompt or "").splitlines()
        if line.strip().startswith(("HEAD:", "POSTURE:", "HANDS:", "FACE:"))
    ]
    return " ".join(lines)


def still_prompt_for(
    emotion: str,
    subject: str | None = SUBJECT_PERSON,
    motion_prompt: str | None = None,
) -> str:
    """Return the image-edit prompt that turns the reference into ``emotion``.

    ``motion_prompt`` is the person's measured motion block (see
    ``src/anubis/utils/motion/motion_prompt.py``); for a still only the
    carriage lines are used, since one frame cannot show a rate.
    """
    family = normalize_reference_subject(subject)
    if family == SUBJECT_NON_HUMAN:
        cue = _NON_HUMAN_CUE_BY_EMOTION.get(emotion)
        if cue is None:
            raise ValueError(f"No still prompt is defined for emotion {emotion!r}.")
        return _NON_HUMAN_STILL_PROMPT_TEMPLATE.format(cue=cue)
    expression = _EXPRESSION_BY_EMOTION.get(emotion)
    if expression is None:
        raise ValueError(f"No still prompt is defined for emotion {emotion!r}.")
    template = (
        _CHARACTER_STILL_PROMPT_TEMPLATE
        if family == SUBJECT_STYLIZED_CHARACTER
        else _PERSON_STILL_PROMPT_TEMPLATE
    )
    posture = _posture_only(motion_prompt)
    motion = _PERSON_STILL_MOTION_TEMPLATE.format(motion_prompt=posture) if posture else ""
    return template.format(expression=expression, motion=motion)


def idle_loop_prompt_for(
    emotion: str,
    subject: str | None = SUBJECT_PERSON,
    motion_prompt: str | None = None,
) -> str:
    """Return the image-to-video prompt for ``emotion``'s idle loop.

    With a ``motion_prompt`` the generic breathe-blink-fidget clause is
    replaced by the person's own measured habits.
    """
    family = normalize_reference_subject(subject)
    label = "neutral, relaxed" if emotion == NEUTRAL_EMOTION else emotion
    if family == SUBJECT_NON_HUMAN:
        motion = _NON_HUMAN_LOOP_MOTION_BY_EMOTION.get(emotion)
        if motion is None:
            raise ValueError(f"No idle loop prompt is defined for emotion {emotion!r}.")
        return _NON_HUMAN_IDLE_LOOP_PROMPT_TEMPLATE.format(emotion=label, motion=motion)
    measured = str(motion_prompt or "").strip()
    if family == SUBJECT_STYLIZED_CHARACTER:
        template = _CHARACTER_IDLE_LOOP_PROMPT_TEMPLATE
        motion_clause = (
            _MEASURED_CHARACTER_IDLE_MOTION_TEMPLATE.format(motion_prompt=measured)
            if measured
            else _GENERIC_CHARACTER_IDLE_MOTION
        )
    else:
        template = _PERSON_IDLE_LOOP_PROMPT_TEMPLATE
        motion_clause = (
            _MEASURED_PERSON_IDLE_MOTION_TEMPLATE.format(motion_prompt=measured)
            if measured
            else _GENERIC_PERSON_IDLE_MOTION
        )
    return template.format(emotion=label, motion=motion_clause)
