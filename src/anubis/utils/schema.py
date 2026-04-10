# src/anubis/utils/classes

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class RouteDecision(BaseModel):
    """"Determine whether to upload media or respond to the conversation. """
    reasoning: str = Field(
        description="Step-by-step reasoning behind the decision for the route."
    )
    route_decision: Literal["chat", "process_media"] = Field(
        description="Classification of the route. chat if responding to the conversation. upload if the user indicates the attached media needs to be added to the identity or uploaded."
    )

class TextualSituationalAwareness(BaseModel):
    classified_situation: Literal["single_speaker", "q_and_a_dialogue", "multi_speaker", "other"]
    reasoning: str = Field(
        description = "Step-by-step reasoning behind the decision for the classified situation of the text. (single speaker monologue, single tweet from user, strictly Q & A, multi-speaker, Other)"
    )

TEXTUAL_SITUATIONAL_AWARENESS_DECISION_INSTRUCTIONS = """
<Role>
Your role is to analyze and classify text with respect to the situation of the content within the text.
</Role>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- A single speaker
- Strictly question and answer between two speakers in a dialogue
- Multiple speakers
- Other 

Present a clear succinct reason why the classification was chosen using examples from the source text to support your reasoning.
</Instructions>

<Rules>
=========== SINGLE SPEAKER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for single speaker situations:

Classify the text as a single speaker given text in the following situations:
- There is a single tweet
- There is a single statement
- There is a label of the speaker and there is only one speaker
- There is only a single speaker detected in the content

Use the following examples to help determine the situation of the given text for single speaker situations:

Example Tweet or single speaker statement:
I believe that through the research and development of A.I., we will understand what is most valuable about being human.

=========== QUESTION AND ANSWER DIALOGUE GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for question and answer dialogue situations:

Classify the text as strictly question and answer between two speakers in a dialogue:
- There is more than one speaker but less than three speakers in the text
- There is turn-taking between two speakers
- There are labels of the speakers and there are only two speakers

=========== MULITPLE SPEAKER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for multiple speaker situations:

Classify the text as multiple speakrs:
- There is more than one speaker but less than three speakers in the text
- There is turn-taking between two speakers
- There are labels of the speakers and there are only two speakers

=========== OTHER GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for situations that do not appropriately match and of the situations for single speakers, question and answer dialogue, or multiple speakers. 

Classify the text as other:
- The criteria for single speaker, question and answer dialogue, and multiple speakers do not apply

Rules for classifying the situation of the text as Other:
- This should not often be used and is treated as an unclassified error and failure in analysis
- This should almost never be used
- There needs to be an explicit reason why this is classified as other, why the single speaker classification does not match, why the question and answer dialogue does not match, why the multiple speaker classification does not match, and each reason needs to be explicitly listed for each reason why the previous classifications did not match the criteria.
- There needs a suggested classification for this situation with an explicit reason why this new suggested classification appropriately classifies the situation of the text better than any of the other reasons.
- Try not to classify any situation of the text as Other.
</Rules>
"""

class MonologuePresentationOrSeriesOfQuotes(BaseModel):          
    """ Determine if this is a fluid train of thought as if in a monologue or presentation for a single speaker or if this is a series of direct quotes from the speaker. """
    classified_situation: Literal["MonologueOrPresentation", "SeriesOfDistinctQuotes"]
    reason: str = Field(description="Step-by-step reasoning behind the decision")

MONOLOGUE_PRESENTATION_OR_SERIES_OF_QUOTES = """
<Role>
Your role is to analyze and classify text with respect to the situation of the content within the text.
</Role>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- A Monologue or Presentation from a single speaker
- Series of Distinct Quotes or Statements from a single speaker
</Instructions>

<Rules>
=========== MONOLOGUE OR PRESENTATION ===========

Use the following rules to help determine the situation of the given text for single speaker situations:

Classify the text as a monologue or presentation given text in the following situation:
- There is a large group of text with a fluid train of thought.
- there is a beginning, middle, and end to the entire body of thought.
- There is a thesis or central topic to the entire body of thought rather than individual lines each with unique topics.
- There is not strictly a list of single lines where each line is a distinct statement.

=========== SERIES OF DISTINCT QUOTES ===========

Use the following rules to help determine the situation of the given text for a series of distinct quotes or statements:

Classify the text as strictly a series of distinct quotes or statements:
- The ideas are not contiuous between newlines
- There is not a fluid train of thought between statements
- Each line is irrelevant completely to the surrounding lines habitually

Example Tweet or single speaker statement:
I believe that through the research and development of A.I., we will understand what is most valuable about being human.
</Rules>
"""

class ProprietaryContentClassification(BaseModel):
           """ Determine if the content is non-personally identifiable such as in a menu or a well known religious text such as the bible or is personally identifiable information such as in an interview, a monoluge, a persentation, or a script. """
           non_personally_identifiable_information: bool = Field(description= "This is TRUE when the text is a menu or well-known religious text such as the bible. This is FALSE otherwise and is intended to be FALSE when there is a script or a monologue or an interview between two people.")
           reasoning: str = Field(
               description = "Step-by-step reasoning behind the decision for the classified situation of the text. (Non-proprietary content includes: single speaker monologue, single tweet from user, strictly Q & A, multi-speaker that is Non-religious. There is a target to identify in the non-religious text document in non-proprietary content.)"
           )


NON_PII_DOCUMENT = """
<Role>
Your role is to analyze and classify text with respect to the situation of the content within the text.
</Role>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- A Non-Personally Identifiable Document
- A Personally Identifiable Document

Present a clear succinct reason why the classification was chosen using examples from the source text to support your reasoning.
</Instructions>

<Rules>
=========== Non-Personally Identifiable Document GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for a Non-Personally Identifiable Document:

Classify the text as a Non-Personally Identifiable Document given text in the following situations:
- The text is a menu for a restaurant
- The text is a religious document such as the Bible or the Koran or other well-known Holy Text.

Use the following examples to help determine the situation of the given text for a Non-Personally Identifiable Document situation:

Example of a menu:
Starbucks Drinks Menu

    Hot Coffee
        Cortado: $5.72, 90 cal
        Brown Sugar Oatmilk Cortado: $6.26, 130 cal
        Caffè Americano: $5.17, 15 cal
        Featured Blonde Single Origin Ethiopia: $3.97, 5 cal
        Featured Medium Roast Pike Place Roast: $3.97, 5 cal
    Iced Tea
        Iced Chai Latte: $6.80, 240 cal
        Iced Green Tea: $4.84, 0 cal
        Iced Peach Green Tea: $5.39, 60 cal
        Starbucks Lemonade: $4.63, 120 cal
    Refreshers
        Cran-Merry Drink: $7.02, 140 cal
        Cran-Merry Orange Lemonade Refresher: $7.02, 140 cal
        Cran-Merry Orange Refresher: $6.48, 100 cal
        Cold Brew: $6.48, 130 cal
        Paradise Drink: $6.48, 140 cal
    Bottled Beverages
        Spindrift Lemon Sparkling Water: $3.26, 0 cal
        Spindrift Raspberry Lime Sparkling Water: $3.26, 9 cal
        Ethos Bottled Water: $3.04, 0 cal
    Cold Coffee
        Pistachio Cream Cold Brew: $6.26, 250 cal
        Salted Caramel Cream Cold Brew: $6.26, 240 cal
        Chocolate Cream Cold Brew: $6.26, 250 cal
        Cold Brew: $5.72, 5 cal
    Cold Tea
        Iced Chai Latte: $6.80, 240 cal
        Iced Green Tea: $4.84, 0 cal
        Iced Peach Green Tea: $5.39, 60 cal
        Starbucks Lemonade: $4.63, 120 cal
    Hot Teas
        Chai Latte: $6.48, 240 cal
        Chai Tea: $4.63, 0 cal
    Frappuccino Blended Beverage
        Pistachio Frappuccino Blended Beverage: $7.35, 380 cal
        Caramel Brûlée Frappuccino Blended Beverage: $7.35, 400 cal
        Peppermint Mocha Frappuccino Blended Beverage: $7.35, 430 cal
    Hot Chocolate, Milk & Juices
        Peppermint Hot Chocolate: $6.80, 440 cal
        Peppermint White Hot Chocolate: $6.80, 480 cal
        White Hot Chocolate: $7.02, 400 cal
        Caramel Apple Spice: $7.02, 380 cal
        Steamed Apple Juice: $4.30, 220 cal

Starbuck-menus.com

# Example of religious text

1:1 In the beginning God created the heaven and the earth.

1:2 And the earth was without form, and void; and darkness was upon
the face of the deep. And the Spirit of God moved upon the face of the
waters.

1:3 And God said, Let there be light: and there was light.

1:4 And God saw the light, that it was good: and God divided the light
from the darkness.

1:5 And God called the light Day, and the darkness he called Night.
And the evening and the morning were the first day.

1:6 And God said, Let there be a firmament in the midst of the waters,
and let it divide the waters from the waters.

1:7 And God made the firmament, and divided the waters which were
under the firmament from the waters which were above the firmament:
and it was so.

1:8 And God called the firmament Heaven. And the evening and the
morning were the second day.

1:9 And God said, Let the waters under the heaven be gathered together
unto one place, and let the dry land appear: and it was so.

1:10 And God called the dry land Earth; and the gathering together of
the waters called he Seas: and God saw that it was good.

=========== Personally Identifiable Document GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for Personally Identifiable Document situations:

Classify the text as a Personally Identifiable Document:
- There is a single tweet
- There is a single statement
- There is a label of the speaker and there is only one speaker
- There is only a single speaker detected in the content

- There is more than one speaker 
- There is turn-taking between two or more speakers
- There are labels of the speakers and the text is NON-RELIGIOUS

Examples of Personally Identifiable Document situations:

@elonmusk I think you're amazing. Thank you for pushing the envelope forward for all of humanity!

Joe Rogan: That's my favorite watch.
Lex Fridman: Thanks Brother.
</Rules>
"""


# ===========================================================
# STEP 2 — Content Situation Classification
# ============================================================
 
class ContentSituationClassification(BaseModel):
    """
    Classifies text into one of five situations so downstream
    routing logic knows how to handle the document.
    """
    classified_situation: Literal[
        "biographical_facts",
        "dialogue",
        "monologue",
        "tweets_or_quotes",
        "presentation",
    ]
    reasoning: str = Field(
        description=(
            "Step-by-step reasoning for the chosen classification. "
            "Cite specific evidence from the text to support the decision."
        )
    )
    has_identifiable_target: bool = Field(
        description=(
            "True when a single named individual is the clear subject or "
            "primary speaker of the text."
        )
    )
    target_name: Optional[str] = Field(
        default=None,
        description=(
            "The full name (or best identifier) of the primary target individual, "
            "if one can be identified. Null otherwise."
        )
    )
 
 
CONTENT_SITUATION_CLASSIFICATION_SYSTEM_PROMPT = """
<Role>
You are an expert content analyst. Your job is to read a body of text and
classify what kind of content it is, so that a downstream data pipeline
knows exactly how to process it.
</Role>
 
<Situations>
Choose exactly ONE of the following classified_situation values:
 
1. biographical_facts
   - Third-person encyclopedic or biographical writing about a person
   - Wikipedia-style articles, "About" pages, press bios, news profiles
   - The subject is described rather than directly speaking
   - No conversational exchange is present
 
2. dialogue
   - Two or more speakers taking turns in a conversation
   - Labelled speakers (e.g. "Alice:", "Bob:") or clearly alternating turns
   - Includes interviews, transcripts, podcasts, screenplays
   - At least two distinct voices are present
 
3. monologue
   - A single speaker delivering a sustained, continuous piece of speech or writing
     in first person
   - Speeches, essays written in first person, blog posts, vlogs, journal entries
   - The speaker is talking about themselves, their ideas, or addressing an audience
 
4. tweets_or_quotes
   - A series of short, discrete, standalone statements
   - Tweets, social-media posts, pull-quotes, aphorisms, one-liners
   - Each statement is self-contained and not part of a flowing conversation
 
5. presentation
   - Structured, formal delivery of information — slides, keynotes, lectures,
     TED talks, product demos — where a single speaker presents to an audience
   - May include audience Q&A but the dominant mode is formal presentation
</Situations>
 
<Instructions>
1. Read the entire text carefully.
2. Identify whether a single individual is the clear subject or speaker.
3. Record that individual's name in target_name (or null if not determinable).
4. Select the single best-matching classified_situation.
5. Write clear, evidence-based reasoning citing the text.
</Instructions>
 
<Rules>
- Choose the classification that best matches the DOMINANT mode of the text.
- If the text is clearly about one person but written by someone else, prefer
  biographical_facts over monologue.
- If in doubt between dialogue and monologue, check whether a second speaker
  actually responds — a rhetorical "question and answer" by one person is still
  a monologue.
- Do NOT invent a target_name. Use only names that appear explicitly in the text.
</Rules>
"""



# ============================================================
# STEP 3 — Conversational → Named Speaker Message Format
# ============================================================
 
class SpeakerMessage(BaseModel):
    speaker: str = Field(
        description=(
            "The speaker's name (e.g. 'Elon Musk') or 'narrator' for "
            "contextual/environmental narration."
        )
    )
    content: str = Field(
        description=(
            "What the speaker said. Physical actions are wrapped in asterisks, "
            "e.g. *smiles warmly* or *pauses to think*."
        )
    )
 
 
class NamedSpeakerMessageFormat(BaseModel):
    """
    Intermediate message format: every turn is attributed to a named speaker
    or 'narrator'. This is the Step 3 output that feeds into Step 4.
    """
    messages: List[SpeakerMessage] = Field(
        description=(
            "Ordered list of speaker turns. Narration and stage directions use "
            "'narrator' as the speaker name."
        )
    )
    identified_speakers: List[str] = Field(
        description="Deduplicated list of every named speaker found in the text (excluding 'narrator')."
    )
 
 
NAMED_SPEAKER_MESSAGE_FORMAT_SYSTEM_PROMPT = """
<Role>
You are a skilled script formatter and dialogue reconstructor. Your job is to
convert any conversational text into a clean, structured list of speaker turns.
</Role>
 
<OutputFormat>
Return a JSON object that matches this schema exactly:
 
{
  "messages": [
    {
      "speaker": "narrator",
      "content": "A stylistic, present-tense description of the setting, atmosphere, or action."
    },
    {
      "speaker": "SPEAKER_NAME",
      "content": "Exactly what the speaker said. Physical actions go in *asterisks*."
    }
  ],
  "identified_speakers": ["Name1", "Name2"]
}
</OutputFormat>
 
<Instructions>
1. Parse the source text and identify every distinct speaker.
2. For each turn, create a message object with the speaker's full name and their content.
3. If there is any contextual framing, scene-setting, or stage direction in the
   original text (or that you must infer to make the exchange coherent), emit a
   narrator message immediately before the relevant turn.
4. Preserve the original wording of each speaker's statements as closely as possible.
5. Render any described physical actions or gestures as *italicised action tags*
   inside the content string (e.g. *leans forward*, *laughs*).
6. Do NOT add invented dialogue. Only infer narrator context where it is clearly
   implied by the source.
7. Record every unique non-narrator speaker in identified_speakers.
</Instructions>
 
<Rules>
- Every message MUST have a "speaker" and a "content" field.
- The narrator speaks only in third-person present tense.
- Speaker names must be consistent throughout (use the same form each time).
- Do not collapse multiple turns from the same speaker into one if they were
  separated by another speaker or a narrator beat.
- If the text is a monologue, there will be exactly one named speaker plus
  optional narrator messages.
- If the text is a series of tweets or quotes, each discrete statement is its
  own message from the same speaker; add a narrator header only if context
  metadata (date, platform, thread) is present.
</Rules>
"""
 

# ============================================================
# STEP 4 — Target Identification + Role Conversion (Best suited for a function)
# ============================================================
 
class RoleMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
 
 
class RoleConvertedMessageFormat(BaseModel):
    """
    Final training-ready format. The target individual becomes 'assistant';
    all other speakers become 'user'. No system message is emitted here —
    the caller injects it separately.
    """
    messages: List[RoleMessage] = Field(
        description=(
            "Ordered role-labelled messages. Target → assistant. "
            "All others → user. Narrator turns are folded into the "
            "immediately following user turn as bracketed context."
        )
    )
    target_name: str = Field(
        description="The name of the individual whose turns are labelled 'assistant'."
    )
    non_target_speakers: List[str] = Field(
        description="Names of all speakers whose turns were converted to 'user'."
    )
 
 
TARGET_IDENTIFICATION_AND_ROLE_CONVERSION_SYSTEM_PROMPT = """
<Role>
You are a training-data formatter. Given:
  (a) a list of named-speaker messages (Step 3 output), and
  (b) the name of the TARGET individual,
your job is to produce the final role-labelled message list for LLM adapter training.
</Role>
 
<TargetInstruction>
The target individual will be identified for you. All turns belonging to that
individual become role = "assistant". Every other named speaker becomes
role = "user". Narrator turns are NOT emitted as standalone messages; instead
their content is prepended to the next user turn in square brackets, e.g.:
  [Scene: a busy coffee shop, mid-morning] What brings you here today?
</TargetInstruction>
 
<OutputFormat>
{
  "messages": [
    { "role": "user",      "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "target_name": "Full Name",
  "non_target_speakers": ["Name1", "Name2"]
}
</OutputFormat>
 
<Rules>
- Do NOT emit a system message — the caller provides that separately.
- Do NOT alter the wording of any speaker's content.
- Preserve turn order exactly.
- If consecutive turns from different non-target speakers occur without a target
  turn between them, concatenate them into a single user message separated by
  a newline, with each attributed in square brackets:
    [Alice] I agree completely.\n[Bob] Me too.
- If consecutive target turns occur without a user turn between them, concatenate
  them into a single assistant message separated by a newline.
- Narrator content always attaches to the NEXT non-narrator turn (user or assistant).
- Return an empty messages list if no target turns are found.
</Rules>
"""
 

 
# ============================================================
# STEP 5a — Target Identification in Conversational Text
# ============================================================
 
class SpeakerStatement(BaseModel):
    speaker: str = Field(description="Name of the speaker for this statement.")
    content: str = Field(description="Verbatim or near-verbatim content of the statement.")
    is_target: bool = Field(description="True if this speaker is the identified target individual.")
 
 
class TargetIdentificationInText(BaseModel):
    """
    Given conversational text, identifies the TARGET individual and labels
    every speaker turn. This is used between Step 2 and Step 3.
    """
    target_name: str = Field(description="Full name (or best identifier) of the target individual.")
    target_identification_reasoning: str = Field(
        description="Why this person was chosen as the target — the primary subject or persona of interest."
    )
    statements: List[SpeakerStatement] = Field(
        description="All speaker turns in document order, each labelled with is_target."
    )
    other_speakers: List[str] = Field(
        description="Names of all other (non-target) speakers found in the text."
    )
 
 
TARGET_IDENTIFICATION_IN_TEXT_SYSTEM_PROMPT = """
<Role>
You are an expert at parsing conversational and multi-speaker text. Given a body
of text and the name of a TARGET individual, your job is to:
  1. Confirm (or best-identify) the target.
  2. Attribute every statement in the text to its speaker.
  3. Flag each statement as belonging to the target or not.
</Role>
 
<Instructions>
You will receive:
  - A body of text (dialogue, transcript, interview, etc.)
  - The name of the TARGET individual (may be approximate or partial)
 
Steps:
1. Scan the text for all speaker labels or contextual attribution cues.
2. Match speaker names to the target (allow for partial matches, nicknames,
   or pronouns clearly referring back to the target).
3. For each discrete statement or turn, create a SpeakerStatement entry with:
     speaker  – the speaker's name as it appears in the text
     content  – the text of the statement (preserve original wording)
     is_target – true only if this speaker is the target individual
4. List all non-target speakers in other_speakers.
5. Explain your target identification reasoning concisely.
</Instructions>
 
<Rules>
- Preserve original statement wording; do not paraphrase.
- If a speaker is ambiguous, use the most likely attribution and note the
  ambiguity in target_identification_reasoning.
- Narrator or stage-direction text should be attributed to speaker = "narrator"
  with is_target = false.
- Do not merge separate turns from the same speaker; each turn is a separate
  SpeakerStatement entry.
- If the target cannot be found in the text, set target_name to "UNKNOWN" and
  explain why in target_identification_reasoning.
</Rules>
"""
 

# ============================================================
# STEP 5b — Characteristic Extraction (Identity Analysis)
# ============================================================
 

"""
characteristic_extraction.py
 
Each dimension of identity analysis is its own Pydantic model + system prompt.
All models follow the same pattern:
  - A `value` field holding the extracted data
  - A `reasoning` field explaining the evidence
  - A `confidence` field (low / medium / high) so callers can gate on quality
  - An `evidence` field quoting or paraphrasing the specific source text
"""
 
# ─────────────────────────────────────────────────────────────
# Shared confidence type
# ─────────────────────────────────────────────────────────────
 
ConfidenceLevel = Literal["low", "medium", "high"]
 
 
# ─────────────────────────────────────────────────────────────
# 1. NAME
# ─────────────────────────────────────────────────────────────
 
class NameExtraction(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description="Full name or best available identifier of the target individual."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="The exact text passage where the name was found or inferred."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence that the extracted name is correct."
    )
    reasoning: str = Field(
        description="Why this name was chosen, including any disambiguation logic."
    )
 
 
NAME_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are an expert text analyst. Extract the full name of the primary target
individual from the provided text.
</Role>
 
<Instructions>
1. Scan the text for explicit name mentions, titles, or clear self-identifications.
2. If multiple names appear, choose the one that is most clearly the subject
   of the document or the primary speaker.
3. Prefer the most complete form of the name (first + last, or known alias).
4. Record the exact passage where you found the name in evidence.
5. Rate your confidence: high = name stated explicitly, medium = strongly implied,
   low = inferred from context only.
</Instructions>
 
<Rules>
- Do NOT invent a name. If no name is detectable, set value to null.
- Do not conflate multiple people. Focus on the single primary target.
- If the text refers to the target only by pronoun, set confidence to low and
  explain in reasoning.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 2. DESCRIPTION
# ─────────────────────────────────────────────────────────────
 
class DescriptionExtraction(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description="A concise one-to-two paragraph summary of who this person is."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Key passages that were synthesised to form this description."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the completeness and accuracy of the description."
    )
    reasoning: str = Field(
        description="How the summary was constructed from the available text."
    )
 
 
DESCRIPTION_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a skilled biographer. Synthesise the source text into a concise,
accurate summary of the target individual.
</Role>
 
<Instructions>
1. Read the entire text.
2. Identify the most important facts, roles, and characteristics of the target.
3. Write a one-to-two paragraph description in third-person present tense.
4. Include: who the person is, what they are known for, and any defining traits.
5. Cite the key passages you drew on in the evidence field.
</Instructions>
 
<Rules>
- Stay strictly within what the text supports. Do not embellish.
- Write in clear, neutral prose — not a list.
- If the text is too sparse for a meaningful description, set value to null
  and explain in reasoning.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 3. IDENTITY
# ─────────────────────────────────────────────────────────────
 
class IdentityExtraction(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description=(
            "How the person defines themselves: profession, role, community, "
            "cultural identity, or other primary self-concept."
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Text passages that reveal how the target self-identifies."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in this identity characterisation."
    )
    reasoning: str = Field(
        description="Explanation of how the identity was inferred from the text."
    )
 
 
IDENTITY_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a sociologist and identity analyst. Determine how the target individual
defines themselves based on the text.
</Role>
 
<Instructions>
1. Look for explicit self-labels: job titles, community affiliations, cultural
   identifiers, political or religious identity statements.
2. Also look for implicit identity signals: what the person talks about most,
   how they frame their own role in the world.
3. Summarise the primary identity in one to three sentences.
4. Quote or closely paraphrase the supporting text in evidence.
</Instructions>
 
<Rules>
- Focus on self-concept, not external labels placed on the person by others.
- If both self-assigned and externally-assigned identities are present,
  note the distinction in reasoning.
- Do not infer identity solely from demographic descriptors unless the target
  explicitly claims them.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 4. HISTORY
# ─────────────────────────────────────────────────────────────
 
class HistoryExtraction(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description=(
            "Key biographical facts, background, and formative events, "
            "presented in roughly chronological order."
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Source passages describing past events or biographical details."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the accuracy and completeness of the history."
    )
    reasoning: str = Field(
        description="How the history was assembled and ordered from the text."
    )
 
 
HISTORY_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a biographical researcher. Extract the life history of the target
individual from the provided text.
</Role>
 
<Instructions>
1. Identify all references to past events, periods, or experiences.
2. Organise them in roughly chronological order.
3. Include: formative experiences, career milestones, educational background,
   notable achievements or failures, and any pivotal turning points.
4. Write as a cohesive narrative paragraph or short bulleted timeline.
5. Record the supporting passages in evidence.
</Instructions>
 
<Rules>
- Include only events that are stated or strongly implied in the text.
- Do not add external biographical knowledge you may have about public figures.
- If dates are absent, note relative ordering (e.g. "before X", "after Y").
- Set value to null if no historical information is available.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 5. EMOTIONS
# ─────────────────────────────────────────────────────────────
 
class EmotionsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Dominant or recurring emotional states expressed or implied "
            "by the target. Each item is a concise label + brief explanation, "
            "e.g. 'Persistent optimism — regularly frames setbacks as growth.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that demonstrate the listed emotional states."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the emotional characterisation."
    )
    reasoning: str = Field(
        description="How each emotional state was inferred from tone, word choice, or explicit statements."
    )
 
 
EMOTIONS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are an expert in affective analysis and emotional intelligence. Identify the
dominant emotional states of the target individual from the text.
</Role>
 
<Instructions>
1. Analyse tone, word choice, explicit emotional statements, and behavioural
   descriptions.
2. Identify recurring or dominant emotions (not one-off reactions to isolated events).
3. For each emotion, write a one-sentence entry: the emotion label followed by
   a dash and a brief textual justification.
4. Capture nuance: distinguish between surface affect and underlying emotional state
   if the text supports this.
5. Populate evidence with the most telling passages.
</Instructions>
 
<Rules>
- Focus on patterns, not single instances, unless the single instance is highly
  significant.
- Use precise emotional vocabulary (e.g. "existential anxiety" rather than just "fear").
- Do not project emotions the text does not support.
- Set value to null if the text contains insufficient emotional signal.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 6. BELIEFS
# ─────────────────────────────────────────────────────────────
 
class BeliefsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Core beliefs or worldview statements. Each item is a declarative "
            "sentence capturing one belief, e.g. 'People are fundamentally good.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that reveal these beliefs."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the belief characterisation."
    )
    reasoning: str = Field(
        description="How each belief was inferred from the text."
    )
 
 
BELIEFS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a philosopher and worldview analyst. Extract the core beliefs of the
target individual from the text.
</Role>
 
<Instructions>
1. Look for statements about how the world works, what is true, what is right,
   or what is possible.
2. Include beliefs about: human nature, society, religion, science, morality,
   or any other domain the target addresses.
3. Distinguish explicitly stated beliefs from strongly implied ones.
4. Express each belief as a clear declarative sentence from the target's
   perspective (first-person implied), e.g. "Hard work always leads to success."
5. Quote supporting evidence.
</Instructions>
 
<Rules>
- Beliefs differ from opinions: beliefs are foundational convictions;
  opinions are specific positions on particular topics.
- Do not conflate the author's framing of the target with the target's own beliefs.
- If a belief appears to have evolved or contradicted itself in the text,
  note both versions in reasoning.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 7. VALUES
# ─────────────────────────────────────────────────────────────
 
class ValuesExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "What the person demonstrably prioritises and protects. These are synonymous with priorities. "
            "Each item names the value and briefly explains how it is demonstrated, "
            "e.g. 'Family — consistently prioritises family time over career advancement.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages where these values are shown in action or statement."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the values characterisation."
    )
    reasoning: str = Field(
        description="How each value was inferred from behaviour, choices, or explicit statements."
    )
 
 
VALUES_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a values analyst and moral psychologist. Identify what the target
individual demonstrably values most from the text.
</Role>
 
<Instructions>
1. Look for what the person protects, sacrifices for, returns to repeatedly,
   or speaks about with the most intensity.
2. Values are shown through behaviour and choices, not just stated preferences.
3. For each value, write: the value name — a one-sentence demonstration from the text.
4. Prioritise values that appear multiple times or in high-stakes contexts.
5. Quote or paraphrase supporting passages.
</Instructions>
 
<Rules>
- Distinguish values from goals: values are enduring principles; goals are
  specific outcomes.
- Do not list generic virtues unless the text specifically supports them
  for this individual.
- Rank by apparent importance if possible (most central first).
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 8. OPINIONS
# ─────────────────────────────────────────────────────────────
 
class OpinionsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Specific stated or strongly implied opinions on concrete topics. "
            "Each item includes the topic and the opinion, "
            "e.g. 'On social media: believes it damages genuine human connection.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages where opinions are expressed."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the opinion attribution."
    )
    reasoning: str = Field(
        description="How each opinion was identified and distinguished from broader beliefs."
    )
 
 
OPINIONS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a discourse analyst. Extract the specific opinions the target individual
holds on concrete topics, as expressed in the text.
</Role>
 
<Instructions>
1. Identify explicit evaluative statements: praise, criticism, endorsement,
   disagreement, or recommendation about specific topics, people, or events.
2. For each opinion, record: the topic + the position held.
3. Note whether the opinion is stated directly or inferred from context.
4. Distinguish strong convictions from tentative or hedged positions.
5. Cite the supporting passages.
</Instructions>
 
<Rules>
- Opinions are specific and situated (on topic X, person Y holds view Z).
  They differ from beliefs (broad worldview) and values (enduring priorities).
- Do not over-generalise a specific opinion into a belief.
- If an opinion appears to have changed across the text, note the evolution.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 9. GOALS
# ─────────────────────────────────────────────────────────────
 
class GoalsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Long-term ambitions and objectives. Each item is a concise statement "
            "of a goal, e.g. 'Build a company that outlasts its founder.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages where goals are stated or clearly implied."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the goal attribution."
    )
    reasoning: str = Field(
        description="How each goal was identified and distinguished from short-term wants."
    )
 
 
GOALS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a motivational analyst. Identify the long-term goals and ambitions of
the target individual from the text.
</Role>
 
<Instructions>
1. Look for explicit statements of aspiration, mission, or long-term intent.
2. Also infer goals from sustained effort, repeated themes, or declared purpose.
3. Distinguish goals (long-term, strategic) from wants (immediate, tactical).
4. Express each goal as a clear, active statement.
5. Cite supporting evidence.
</Instructions>
 
<Rules>
- A goal implies a future state the person is actively working toward.
- Do not include vague aspirations unless they are anchored to specific intentions
  in the text.
- If a goal is explicitly stated as already achieved, note that it was a past goal.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 10. WANTS
# ─────────────────────────────────────────────────────────────
 
class WantsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Immediate desires or things the person is actively pursuing right now. "
            "e.g. 'Wants public recognition for recent work.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages where immediate wants are expressed."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the wants characterisation."
    )
    reasoning: str = Field(
        description="How each want was distinguished from deeper needs or long-term goals."
    )
 
 
WANTS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a behavioural analyst. Identify what the target individual wants
right now — their immediate desires and active pursuits — from the text.
</Role>
 
<Instructions>
1. Look for present-tense desires, requests, complaints, or expressed cravings.
2. Wants are immediate and specific (e.g. "wants more money now") as opposed to
   long-term goals ("wants to be financially independent").
3. For each want, write a concise statement.
4. Cite the supporting passages.
</Instructions>
 
<Rules>
- Wants are surface-level and often transient; needs are deeper and more stable.
- Do not conflate a stated preference with a deep need unless the text makes
  the deeper significance clear.
- Include wants the person expresses frustration about not yet having.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 11. NEEDS
# ─────────────────────────────────────────────────────────────
 
class NeedsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Deeper psychological or practical needs, stated or implied. "
            "e.g. 'Needs external validation to feel secure in decisions.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that reveal underlying needs."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the needs characterisation."
    )
    reasoning: str = Field(
        description="How each need was inferred, especially when not explicitly stated."
    )
 
 
NEEDS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a depth psychologist. Identify the deeper, often unspoken needs of
the target individual from the text.
</Role>
 
<Instructions>
1. Look beneath stated wants to identify underlying psychological or practical needs
   (e.g. safety, belonging, autonomy, esteem, meaning, control).
2. Use Maslow's hierarchy, self-determination theory, or similar frameworks
   as analytical lenses — but ground every claim in textual evidence.
3. For each need, write a concise statement and explain what in the text
   reveals it (often indirectly).
4. Cite supporting passages.
</Instructions>
 
<Rules>
- Needs are often implicit. Justify inferences carefully.
- A need can be inferred from patterns of behaviour, emotional reactions,
  or what the person repeatedly seeks or avoids.
- Do not speculate beyond what the text supports. Rate confidence honestly.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 12. FEARS
# ─────────────────────────────────────────────────────────────
 
class FearsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Things the person is afraid of or actively avoids. "
            "e.g. 'Fear of irrelevance — avoids topics where their authority could be challenged.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that reveal fears, avoidance, or anxiety."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the fear attribution."
    )
    reasoning: str = Field(
        description="How each fear was identified, including whether it is stated or inferred."
    )
 
 
FEARS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a clinical narrative analyst. Identify what the target individual
fears or actively avoids, as revealed in the text.
</Role>
 
<Instructions>
1. Look for explicit statements of fear, dread, or anxiety.
2. Also look for avoidance behaviours, defensive reactions, or topics the
   person deflects or minimises.
3. Distinguish existential fears (fear of death, meaninglessness) from
   situational fears (fear of a specific outcome or person).
4. For each fear, write: the fear + a brief explanation of how it manifests.
5. Cite supporting passages.
</Instructions>
 
<Rules>
- Fears may be disguised as anger, avoidance, or over-confidence — look for
  these masks.
- Do not diagnose. Describe patterns, not clinical conditions.
- Rate confidence carefully: stated fears = high; inferred = medium or low.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 13. FLAWS
# ─────────────────────────────────────────────────────────────
 
class FlawsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Acknowledged or observable weaknesses, blind spots, or contradictions. "
            "e.g. 'Impatience — frequently interrupts others and rushes decisions.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that reveal flaws or contradictions."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the flaw attribution."
    )
    reasoning: str = Field(
        description="How each flaw was identified and supported by the text."
    )
 
 
FLAWS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a character analyst. Identify the flaws, weaknesses, and blind spots
of the target individual from the text.
</Role>
 
<Instructions>
1. Look for acknowledged self-criticisms, repeated mistakes, contradictions
   between stated values and actual behaviour, or observations by others.
2. Include cognitive biases, emotional blind spots, interpersonal patterns,
   or habitual errors in judgment.
3. For each flaw, write a concise label + a one-sentence description of how
   it appears in the text.
4. Distinguish between flaws the person is aware of and those they appear
   unaware of.
5. Cite supporting passages.
</Instructions>
 
<Rules>
- Be specific and evidence-based. "Arrogance" only qualifies if the text
  shows it clearly.
- Do not conflate cultural or stylistic differences with moral flaws.
- Note whether the flaw is self-reported or observed by others.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 14. STRENGTHS
# ─────────────────────────────────────────────────────────────
 
class StrengthsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Demonstrated strengths, skills, or exceptional qualities. "
            "e.g. 'Strategic clarity — consistently identifies the core issue quickly.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that demonstrate these strengths."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the strengths characterisation."
    )
    reasoning: str = Field(
        description="How each strength was identified from the text."
    )
 
 
STRENGTHS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a strengths-based psychologist and talent analyst. Identify the
demonstrated strengths and exceptional qualities of the target individual
from the text.
</Role>
 
<Instructions>
1. Look for repeated demonstrations of skill, excellence, or positive impact.
2. Include: cognitive strengths (e.g. analytical thinking), interpersonal
   strengths (e.g. empathy), creative strengths, leadership strengths,
   emotional strengths (e.g. resilience), and domain expertise.
3. For each strength, write: the strength label — a one-sentence explanation
   of how it is demonstrated.
4. Prefer strengths that appear multiple times or in high-stakes contexts.
5. Cite supporting passages.
</Instructions>
 
<Rules>
- Strengths must be demonstrated, not merely claimed by the target.
- If others praise the target for something, it qualifies as evidence.
- Do not list generic positives without specific textual support.
- Distinguish consistent strengths from one-time achievements.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 15. PROBLEMS
# ─────────────────────────────────────────────────────────────
 
class ProblemsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Active problems, challenges, conflicts, or obstacles the target "
            "is facing or has faced. "
            "e.g. 'Strained relationship with co-founder creating organisational tension.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages that reveal these problems."
    )
    confidence: ConfidenceLevel = Field(
        description="Confidence in the problem identification."
    )
    reasoning: str = Field(
        description="How each problem was identified and whether it is current or past."
    )
 
 
PROBLEMS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a situational analyst. Identify the active problems, challenges,
conflicts, and obstacles facing the target individual, as described in the text.
</Role>
 
<Instructions>
1. Look for explicit problems, crises, conflicts, complaints, or obstacles.
2. Include: interpersonal conflicts, professional challenges, internal struggles,
   systemic barriers, or unresolved tensions.
3. For each problem, write a concise statement describing what the problem is
   and who or what it involves.
4. Note whether the problem is current, recurring, or resolved by the end of
   the text.
5. Cite supporting passages.
</Instructions>
 
<Rules>
- Problems differ from flaws: problems are external or situational challenges;
  flaws are internal character issues (though a flaw may cause a problem).
- Do not speculate about problems not evidenced in the text.
- If a problem appears resolved, mark it as past in your reasoning.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 16. RELATIONSHIPS
# ─────────────────────────────────────────────────────────────
 
class RelationshipEntry(BaseModel):
    person: str = Field(description="Name of the other individual.")
    relationship_type: str = Field(
        description="e.g. 'close friend', 'rival', 'mentor', 'romantic partner', 'colleague'."
    )
    dynamic: str = Field(
        description="One-to-two sentence description of the relationship dynamic."
    )
    sentiment: Literal["positive", "negative", "ambivalent", "neutral"] = Field(
        description="The target's apparent emotional orientation toward this person."
    )
 
 
class RelationshipsExtraction(BaseModel):
    value: Optional[List[RelationshipEntry]] = Field(
        default=None,
        description="Named individuals and the nature of their relationship to the target."
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Passages describing or implying these relationships."
    )
    confidence: ConfidenceLevel = Field(
        description="Overall confidence in the relationship characterisation."
    )
    reasoning: str = Field(
        description="How each relationship and its dynamic were inferred from the text."
    )
 
 
RELATIONSHIPS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are a relational analyst and social psychologist. Map the significant
relationships of the target individual from the text.
</Role>
 
<Instructions>
1. Identify every named individual who has a meaningful relationship with
   the target.
2. For each person, determine:
     person           – their name
     relationship_type – the structural role (friend, rival, mentor, etc.)
     dynamic          – a 1-2 sentence description of how they interact
     sentiment        – the target's apparent emotional orientation (positive /
                        negative / ambivalent / neutral)
3. Cite the specific passages that reveal each relationship.
</Instructions>
 
<Rules>
- Only include named individuals. Do not include unnamed or generic groups
  unless they are clearly described as a specific identifiable unit.
- Base relationship_type and sentiment on the target's perspective, not
  the other person's.
- If the relationship dynamic has shifted within the text, describe the arc.
- Do not infer romantic or adversarial relationships without textual support.
</Rules>
"""
 
 
# ─────────────────────────────────────────────────────────────
# 17. SECRETS
# ─────────────────────────────────────────────────────────────
 
class SecretsExtraction(BaseModel):
    value: Optional[List[str]] = Field(
        default=None,
        description=(
            "Information the target holds privately and does not wish to expose "
            "or share with others, revealed only through trust or confidential "
            "disclosure. Each item is a concise statement of the secret or "
            "category of concealed information, "
            "e.g. 'Conceals a past failure that contradicts their public narrative.'"
        )
    )
    evidence: Optional[str] = Field(
        default=None,
        description=(
            "Passages — including omissions, deflections, contradictions, or "
            "confidential disclosures — that hint at or reveal withheld information."
        )
    )
    confidence: ConfidenceLevel = Field(
        description=(
            "Confidence in the secret attribution. "
            "high = explicitly disclosed in confidence; "
            "medium = strongly implied by contradiction or avoidance; "
            "low = speculative inference."
        )
    )
    reasoning: str = Field(
        description=(
            "Detailed explanation of how the secret was detected — whether through "
            "direct confidential disclosure, behavioural contradiction, omission, "
            "evasion, or indirect reference."
        )
    )
 
 
SECRETS_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are an expert in subtext analysis, narrative psychology, and confidential
disclosure patterns. Your task is to identify information that the target
individual holds privately — things they conceal, withhold, or share only
under conditions of deep trust — as revealed by the provided text.
</Role>
 
<WhatCountsAsASecret>
A secret is any of the following:
  1. Information the target EXPLICITLY discloses as private, confidential,
     or shared only with trusted individuals.
  2. Information the target ACTIVELY conceals, deflects from, or contradicts
     when it is raised — the concealment itself is evidence.
  3. A past event, belief, identity, or action that is INCONSISTENT with the
     target's public persona and appears to be deliberately suppressed.
  4. A desire, fear, or opinion the target expresses ONLY in private contexts
     or to specific trusted individuals, as contrasted with their public
     statements.
</WhatCountsAsASecret>
 
<Instructions>
1. Read the text for explicit confidential disclosures ("I've never told
   anyone this…", "between us…", "I don't talk about this publicly…").
2. Look for CONTRADICTIONS between stated public identity and private behaviour
   or off-the-record statements — the gap is often where secrets live.
3. Notice topics the target AVOIDS, deflects, laughs off, or answers
   incompletely when directly asked.
4. Note OMISSIONS — things that would logically be mentioned but are not.
5. For each secret or category of concealed information, write a concise
   one-sentence entry.
6. Record the textual evidence (including the absence or evasion) in the
   evidence field.
7. Rate confidence carefully:
     high   = target explicitly disclosed this as private/secret
     medium = strong contradiction or deliberate avoidance pattern
     low    = speculative inference from limited signals
</Instructions>
 
<Rules>
- Do NOT fabricate secrets. Every entry must be grounded in textual signals.
- A secret does not have to be shameful — it may simply be private (e.g.
  a deeply held aspiration not shared publicly).
- Distinguish secrets the target is aware they are keeping from information
  the target appears unconsciously to suppress (note the distinction in
  reasoning).
- Do not conflate privacy with deception — a person may keep secrets without
  being dishonest.
- If no secrets or confidential disclosures are detectable, set value to null
  and explain in reasoning.
- Respect the sensitivity of this field: describe the category or nature of
  the secret rather than sensationalising it.
</Rules>
"""
 

 
# ─────────────────────────────────────────────────────────────
# Master registry — maps dimension name → (model class, system prompt)
# Canonical order matches GeneralCharacteristicExtraction field order.
# Useful for iterating all analyses programmatically.
# ─────────────────────────────────────────────────────────────
 
CHARACTERISTIC_EXTRACTORS: dict[str, tuple[type[BaseModel], str]] = {
    "name":          (NameExtraction,          NAME_EXTRACTION_SYSTEM_PROMPT),
    "description":   (DescriptionExtraction,   DESCRIPTION_EXTRACTION_SYSTEM_PROMPT),
    "identity":      (IdentityExtraction,      IDENTITY_EXTRACTION_SYSTEM_PROMPT),
    "history":       (HistoryExtraction,       HISTORY_EXTRACTION_SYSTEM_PROMPT),
    "emotions":      (EmotionsExtraction,      EMOTIONS_EXTRACTION_SYSTEM_PROMPT),
    "beliefs":       (BeliefsExtraction,       BELIEFS_EXTRACTION_SYSTEM_PROMPT),
    "values":        (ValuesExtraction,        VALUES_EXTRACTION_SYSTEM_PROMPT),
    "opinions":      (OpinionsExtraction,      OPINIONS_EXTRACTION_SYSTEM_PROMPT),
    "goals":         (GoalsExtraction,         GOALS_EXTRACTION_SYSTEM_PROMPT),
    "wants":         (WantsExtraction,         WANTS_EXTRACTION_SYSTEM_PROMPT),
    "needs":         (NeedsExtraction,         NEEDS_EXTRACTION_SYSTEM_PROMPT),
    "fears":         (FearsExtraction,         FEARS_EXTRACTION_SYSTEM_PROMPT),
    "problems":      (ProblemsExtraction,      PROBLEMS_EXTRACTION_SYSTEM_PROMPT),
    "flaws":         (FlawsExtraction,         FLAWS_EXTRACTION_SYSTEM_PROMPT),
    "strengths":     (StrengthsExtraction,     STRENGTHS_EXTRACTION_SYSTEM_PROMPT),
    "secrets":       (SecretsExtraction,       SECRETS_EXTRACTION_SYSTEM_PROMPT),
    "relationships": (RelationshipsExtraction, RELATIONSHIPS_EXTRACTION_SYSTEM_PROMPT),
}
 
 


class Relationship(BaseModel):
    person: str
    relationship_type: str = Field(description="e.g. 'close friend', 'rival', 'mentor', 'romantic partner'")
    description: str = Field(description="Brief description of the dynamic between the target and this person.")
 
 
# ─────────────────────────────────────────────────────────────
# GeneralCharacteristicExtraction
# Aggregate model — all 17 dimensions in one structured output.
# Use this when you want a single model call to populate every
# dimension at once instead of running CHARACTERISTIC_EXTRACTORS
# individually.
# ─────────────────────────────────────────────────────────────
 
class GeneralCharacteristicExtraction(BaseModel):
    """
    Rich identity profile inferred from a body of text about or by a target.
    All fields are optional — only populate what can be reasonably inferred.
    Run the individual *Extraction models for higher-fidelity per-dimension
    analysis; use this model for a single-pass broad profile.
    """
    name: Optional[str] = Field(
        default=None,
        description="Full name of the target individual."
    )
    description: Optional[str] = Field(
        default=None,
        description="One-paragraph summary of who this person is."
    )
    identity: Optional[str] = Field(
        default=None,
        description="How the person identifies themselves (profession, role, community, etc.)."
    )
    history: Optional[str] = Field(
        default=None,
        description="Key biographical facts, background, formative events."
    )
    emotions: Optional[List[str]] = Field(
        default=None,
        description="Dominant or recurring emotional states expressed or implied."
    )
    beliefs: Optional[List[str]] = Field(
        default=None,
        description="Core beliefs or worldview statements inferred from the text."
    )
    values: Optional[List[str]] = Field(
        default=None,
        description="What the person demonstrably cares about most."
    )
    opinions: Optional[List[str]] = Field(
        default=None,
        description="Specific stated or strongly implied opinions on topics."
    )
    goals: Optional[List[str]] = Field(
        default=None,
        description="Long-term ambitions or objectives."
    )
    wants: Optional[List[str]] = Field(
        default=None,
        description="Immediate desires or things the person is actively pursuing."
    )
    needs: Optional[List[str]] = Field(
        default=None,
        description="Deeper psychological or practical needs, stated or implied."
    )
    fears: Optional[List[str]] = Field(
        default=None,
        description="Things the person is afraid of or actively avoids."
    )
    problems: Optional[List[str]] = Field(
        default=None,
        description=(
            "Current challenges the individual is facing. These could be internal "
            "conflicts or external conflicts that the individual needs to resolve."
        )
    )
    flaws: Optional[List[str]] = Field(
        default=None,
        description="Acknowledged or observable weaknesses, blind spots, or contradictions."
    )
    strengths: Optional[List[str]] = Field(
        default=None,
        description=(
            "Features of the individual that the individual excels at. The best "
            "qualities of the individual and what the individual is best at doing "
            "or what is best about the individual."
        )
    )
    secrets: Optional[List[str]] = Field(
        default=None,
        description=(
            "Information held privately by the individual that they do not wish to "
            "expose or share with others unless they trust and confidently confide "
            "in someone."
        )
    )
    relationships: Optional[List[RelationshipEntry]] = Field(
        default=None,
        description="Named individuals and the nature of their relationship to the target."
    )
    reasoning: str = Field(
        description=(
            "Step-by-step explanation of how each populated field was inferred "
            "from the source text, with cited evidence for every claim."
        )
    )
 
 
GENERAL_CHARACTERISTIC_EXTRACTION_SYSTEM_PROMPT = """
<Role>
You are an expert psychologist, biographer, and narrative analyst. Your task is
to build a rich identity profile of a TARGET individual from a body of text.
The text may be written BY the target (first-person) or ABOUT the target (third-person),
or a combination of both.
</Role>
 
<Instructions>
Carefully read the provided text and extract or infer as much as possible about
the target individual across the following dimensions:
 
  name          – Full name or best identifier.
  description   – A concise summary paragraph.
  identity      – How the person defines themselves (job, role, community).
  history       – Key biographical facts and formative experiences.
  emotions      – Dominant emotional states, mood, affect.
  beliefs       – Core convictions about the world, people, or themselves.
  values        – What they demonstrably prioritise and protect.
  opinions      – Specific viewpoints on concrete topics.
  goals         – Long-term ambitions and objectives.
  wants         – Immediate desires or active pursuits.
  needs         – Deeper psychological or material needs.
  fears         – What the person avoids, dreads, or is threatened by.
  problems      – Current challenges the individual is facing; internal or
                  external conflicts the individual needs to resolve.
  flaws         – Acknowledged or implied weaknesses and contradictions.
  strengths     – The features the individual excels at; their best qualities
                  and what they are best at doing or what is best about them.
  secrets       – Information held privately that the individual does not wish
                  to expose or share with others unless they trust and
                  confidently confide in someone. Detected through explicit
                  confidential disclosure, contradiction between public and
                  private behaviour, deliberate avoidance, or notable omission.
  relationships – Named individuals and the nature of their connection to the
                  target, including relationship type, dynamic, and the target's
                  emotional orientation toward them.
 
For each field, cite or paraphrase the specific evidence from the text that
led to your inference.
</Instructions>
 
<DimensionGuidance>
 
PROBLEMS vs FLAWS
  problems  = external or situational challenges the person must navigate
              (a conflict with a colleague, a failing project, a health crisis).
  flaws     = internal character weaknesses or habitual errors in judgment
              (impatience, self-sabotage, confirmation bias).
  A flaw may cause a problem, but they are distinct dimensions.
 
STRENGTHS vs VALUES
  strengths = what the person is demonstrably good at (skills, capacities).
  values    = what the person cares about and prioritises (principles, commitments).
  A person may value honesty but not be particularly strong at delivering it.
 
WANTS vs NEEDS vs GOALS
  wants  = immediate, surface-level desires ("I want more recognition now").
  needs  = deeper psychological or material requirements ("needs belonging").
  goals  = long-term, strategic ambitions ("build a lasting institution").
 
SECRETS
  A secret is any information the target:
    (a) explicitly discloses as private or confidential,
    (b) actively conceals, deflects from, or contradicts,
    (c) omits in contexts where it would logically be mentioned, or
    (d) expresses only in private/trusted contexts vs public statements.
  Rate confidence: high = explicit private disclosure; medium = strong
  contradiction or avoidance pattern; low = speculative inference.
 
BELIEFS vs OPINIONS
  beliefs  = foundational convictions about how the world is or should be.
  opinions = specific, situated positions on concrete topics or events.
 
</DimensionGuidance>
 
<Rules>
- Only populate a field if you have reasonable textual evidence for it.
- Do NOT invent or hallucinate details.
- Prefer direct quotations or close paraphrase as evidence.
- If the text contains multiple conflicting signals for a field, list both and
  note the conflict in your reasoning.
- Keep list items concise (one sentence each where possible).
- relationships entries must name a real person mentioned in the text.
- For secrets, describe the category or nature of the withheld information
  rather than sensationalising it.
- Always populate the reasoning field with your full chain of thought,
  citing the specific textual evidence for each populated dimension.
</Rules>
"""
 