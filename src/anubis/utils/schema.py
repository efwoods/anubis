# src/anubis/utils/classes

from __future__ import annotations
from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Annotated, Sequence
from dataclasses import dataclass, field

import operator


class ModelResponseMetricMetadata:
    """Represents a standardized set of metrics for a specific model type."""

    def __init__(self):
        self.latency_list_ms: list[float] = []
        self.calls_count: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.average_latency_ms: float = 0.0

    async def update_metrics(self, response) -> None:
        """Update the metrics with the response from the model."""
        self.latency_list_ms.append(response.get("latency_ms", 0.0))
        self.calls_count += 1
        self.prompt_tokens += response.get("token_usage", {}).get("prompt_tokens", 0)
        self.completion_tokens += response.get("token_usage", {}).get(
            "completion_tokens", 0
        )
        self.total_tokens += response.get("token_usage", {}).get("total_tokens", 0)
        self.total_cost += response.get("total_cost", 0.0)

        if self.latency_list_ms:
            self.average_latency_ms = sum(self.latency_list_ms) / len(
                self.latency_list_ms
            )

    async def to_dict(self) -> dict:
        """Returns the metrics as a dictionary for serialization."""
        return {
            "latency_list_ms": self.latency_list_ms,
            "calls_count": self.calls_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_latency_ms": self.average_latency_ms,
        }


class RouteDecision(BaseModel):
    """Determine whether to upload media or respond to the conversation."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the decision for the route."
    )
    route_decision: Literal["chat", "process_media"] = Field(
        description="Classification of the route. chat if responding to the conversation. upload if the user indicates the attached media needs to be added to the identity or uploaded."
    )


class TextualSituationalAwareness(BaseModel):
    classified_situation: Literal[
        "single_speaker", "q_and_a_dialogue", "multi_speaker", "other"
    ]
    reasoning: str = Field(
        description="Step-by-step reasoning behind the decision for the classified situation of the text. (single speaker monologue, single tweet from user, strictly Q & A, multi-speaker, Other)"
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
    """Determine if this is a fluid train of thought as if in a monologue or presentation for a single speaker or if this is a series of direct quotes from the speaker."""

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


class ReferenceDocumentOrBiographicalConversationalInformation(BaseModel):
    """Determine if the content is one of two categories: the first category is a reference document such as a menu or a well known religious text such as the bible. The second category is conversational or biographical information such as in an interview, a monologue, a persentation, biography, or a script."""

    is_menu_or_religious_text: bool = Field(
        description="""This is TRUE when the text is a menu or well-known religious text such as the bible. This is FALSE otherwise and is intended to be FALSE when there is a script or a monologue or an interview between two people. This is FALSE when there is a biography of a real or fictional person. If the reasoning determines this is FALSE, then is_reference_document needs to match the reasoning determination exactly. TRUE only when the text is a restaurant/food menu OR a well-known religious holy text (e.g. Bible, Quran, Torah). FALSE for everything else — including wiki pages, character biopages, interviews, monologues, biographies, scripts, tweets, and Q&A. A formal or encyclopedic writing style does NOT make a document a reference document. IMPORTANT: this value MUST match the conclusion in your reasoning field. If reasoning says FALSE, this must be FALSE."""
    )
    reasoning: str = Field(
        description="Step-by-step reasoning behind the decision for the classified situation of the text. (Biographical conversational informaiton includes: single speaker monologue, single tweet from user, a biography, strictly Question and answer, multi-speaker content that is Non-religious. There is a target to identify in the non-religious text document in biographical conversational informaiton when is_reference_document is FALSE.)"
    )


REFERENCE_DOCUMENT_OR_BIOGRAPHICAL_CONVERSATIONAL_INFORMATION = """
<Role>
Your role is to analyze and classify text with respect to the situation of the content within the text.
</Role>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- Menu or Religious Document
- A Biographical or Conversational Document

Present a clear succinct reason why the classification was chosen using examples from the source text to support your reasoning.
</Instructions>

<Rules>
=========== Non-biographical or Conversational Document GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for a Non-biographical or Conversational Document:

Classify the text as a Non-biographical or Conversational Document given text in the following situations:
- The text is a menu for a restaurant
- The text is a religious document such as the Bible or the Koran or other well-known Holy Text.

Use the following examples to help determine the situation of the given text for a Non-biographical or Conversational Document situation:

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

=========== Biographical or Conversational Document GUIDELINES FOLLOW ===========

Use the following rules to help determine the situation of the given text for Biographical or Conversational Document situations:

Classify the text as a Biographical or Conversational Document:
- There is a single tweet
- There is a single statement
- There is a label of the speaker and there is only one speaker
- There is only a single speaker detected in the content
- There is biographical information about a real person or fictional character.

- There is more than one speaker 
- There is turn-taking between two or more speakers
- There are labels of the speakers and the text is NON-RELIGIOUS

IMPORTANT
- Wiki articles or encyclopedia-style pages about characters, people, or topics
- Pages from fandom/fan wikis (e.g. Narutopedia, Wookieepedia)
- Biographical pages about real or fictional persons, regardless of writing style
- Any formal or structured document that is NOT a menu or religious text
</Rules>


<EXAMPLE>
Examples of Biographical or Conversational Document situations:

@elonmusk I think you're amazing. Thank you for pushing the envelope forward for all of humanity!
</EXAMPLE>

<EXAMPLE>
Examples of Biographical or Conversational Document situations:
Joe Rogan: That's my favorite watch.
Lex Fridman: Thanks Brother.
</EXAMPLE>

<EXAMPLE>
Examples of Biographical or Conversational Document situations:
The given text is a wiki page about the character Kurama from the Naruto series. It contains detailed information about the character\'s background, personality, appearance, abilities, and other relevant details. The text is written in a formal and informative style, typical of a wiki article
</EXAMPLE>

<Instructions>
Your objective is the following:
Classify the text and decide whether the text contains one of the following situations:
- Menu or Religious Document
- A Biographical or Conversational Document

Present a clear succinct reason why the classification was chosen using examples from the source text to support your reasoning.
</Instructions>


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
    
        description=(
            "The full name (or best identifier) of the primary target individual, "
            "if one can be identified. Null otherwise."
        )
    )


CONTENT_SITUATION_CLASSIFICATION_SYSTEM_PROMPT = """
# Role and Objective

You are an expert content analyst. Your job is to read a body of text and classify what kind of content it is, so that a downstream data pipeline knows exactly how to process it.

You must produce exactly ONE `classified_situation` value, identify whether a single named individual is the primary subject or speaker, and provide clear evidence-based reasoning that cites the text.

# Instructions

- Read the entire text before deciding.
- Choose exactly ONE `classified_situation` from: `biographical_facts`, `dialogue`, `tweets_or_quotes`, or `monologue`.
- Choose the classification that best matches the DOMINANT mode of the text.
- Determine whether a single individual is the clear subject or speaker, and record that individual's name in `target_name` (or null if not determinable).
- Write reasoning that cites specific evidence from the source text.
- Do NOT invent a `target_name`. Use only names that appear explicitly in the text.

## STRONG PRIOR — default to a single speaker when labels are absent

This is the most important rule and OVERRIDES superficial cues like line breaks, tone shifts, name mentions, or topic changes.

- A new line, a paragraph break, a topic shift, an emotional shift, a punctuation change, or a name being mentioned does NOT indicate a speaker change.
- Each line in a line-separated body of text is, by default, a DISTINCT but DISJOINTED statement from the SAME single speaker.
- Treat the text as authored by a single speaker UNLESS the text contains two or more EXPLICIT speaker labels (see definition below) that each attach to at least one distinct statement.
- If you are uncertain, the abduction MUST lean toward a single-speaker classification (`tweets_or_quotes`, `monologue`, or `biographical_facts`) — NOT `dialogue`.

### Definition — what counts as an EXPLICIT speaker label

An explicit speaker label is a structural marker that unambiguously attributes a span of text to a specific speaking entity. ONLY the following count as explicit speaker labels:

- An inline name-prefix attached to a statement, on the same line, followed by a colon or a dash (e.g. `Alice: Hi there.` / `Bob - I disagree.`).
- A screenplay-style or interview-style label on its own line that introduces the next utterance (e.g. `INTERVIEWER:` followed by a quoted answer).
- A structured `speaker`, `from`, `author`, or equivalent field in a JSON/structured transcript that is paired with a corresponding `text` / `content` field for that turn.

The following do NOT count as explicit speaker labels, even when present:
- A name appearing inside the body of a statement (e.g. "Tell Alice I said hi.").
- An @-mention inside the text (e.g. `@elonmusk great point`).
- Message-app metadata such as iMessage, SMS, or chat-export style headers when not paired one-to-one with the lines that follow.
- Hashtags, signatures, salutations, or sign-offs.
- Different topics, different tones, different timestamps, or different line breaks alone.

## Sub-categories for more detailed instructions

### Category 1 — `biographical_facts`
- Third-person encyclopedic or biographical writing about a person.
- Wikipedia-style articles, "About" pages, press bios, news profiles.
- The subject is described rather than directly speaking.
- No conversational exchange is present.
- If the text is clearly about one person but written by someone else, prefer `biographical_facts` over `monologue`.

### Category 2 — `dialogue`

`dialogue` is a HIGH-BAR classification. ALL of the following hard requirements must be satisfied; if ANY one of them fails, the text is NOT a dialogue.

Hard requirements (all must be true):
1. The text contains TWO OR MORE explicit speaker labels (per the definition above) for distinct speaking entities.
2. EACH of those labeled speakers must produce at least one distinct statement that is attributed to them by an explicit label.
3. The labeled statements must form turns — i.e. one labeled speaker's utterance is followed by another labeled speaker's utterance.

Additional clarifications:
- Includes interviews, transcripts, podcasts, screenplays.
- At least two distinct voices are present.
- There will be clearly labeled speakers.
- The content may be a string of JSON with labeled speakers.
- A dialogue MUST contain at least two distinct conversational statements from two distinct speakers.
- A single statement from a single speaker mentioning multiple names does NOT constitute a dialogue.
- Lines of text that are not contiguous in meaning (no thesis between the statements and no logical flow between ideas between statements) do NOT constitute a dialogue.
- The presence of varying names, @-mentions, hashtags, or message-style indicators (e.g., iMessage, SMS) does NOT, on its own, show a back-and-forth conversation. If each new line is a distinct statement and there are no explicit speaker labels attaching different lines to different speakers, THIS IS NOT A DIALOGUE — classify as `tweets_or_quotes` (or `monologue` if the lines flow as one connected piece).
- If any of the hard requirements above is in doubt, do NOT classify as `dialogue`. Default to a single-speaker classification.

### Category 3 — `tweets_or_quotes`

This is the category for line-separated text from a single speaker when the lines are disjointed and unlabled where each line is a distinct, disjointed statement from the same speaker.

- A series of short, discrete, standalone statements that are considered spoken words directly from the target speaker.
- Tweets, social-media posts, pull-quotes, aphorisms, one-liners, status updates, chat messages from one person.
- Each statement is self-contained and not part of a flowing conversation.
- Each line is a DISTINCT, DISJOINTED statement from the SAME speaker — even when the topic, tone, recipient, or emotion changes between lines.
- If there is a direct object in the text yet no response from another speaker and the text is a list of strings and not JSON, classify as `tweets_or_quotes`.
- If there is a list of distinct statements with no explicit speaker labels and the text is not contiguous in meaning between lines (no thesis or logical flow between statements), classify as `tweets_or_quotes`.
- A line containing an @-mention, a name, or a reply-style fragment is still attributed to the SAME single speaker unless an explicit speaker label says otherwise.

### Category 4 — `monologue`
- One long body of text originating from a single speaker (e.g. a presentation, a speech, a lecture, or similar content).
- Structured, formal delivery of information — slides, keynotes, lectures, TED talks, product demos — where a single speaker presents to an audience may be present.
- A single speaker delivering a sustained, continuous piece of speech or writing in first person.
- Speeches, essays written in first person, blog posts, vlogs, and journal entries are examples of a monologue.
- The speaker is talking about themselves, their ideas, addressing an audience, or speaking about a topic, subject, or idea.
- The content may be a string of JSON with labeled speakers.
- There will only be a single speaker. There may be an introduction to the speaker and a conclusion to the speaker, but there will be no response from another speaker.
- There will NOT be Question and Answer between the speaker and the audience. Question and Answer when present is a dialogue.
- If in doubt between `dialogue` and `monologue`, check whether a second speaker actually responds — a rhetorical "question and answer" by one person is still a monologue.

# Reasoning Steps

1. Read the entire text carefully.
2. SCAN for explicit speaker labels (per the definition in the Instructions). Count how many distinct speaker labels exist AND how many of them have at least one statement directly attributed to them.
   - If fewer than two distinct labeled entities each produce at least one labeled statement, `dialogue` is ELIMINATED. Move on without considering it further.
3. Identify how many distinct speakers actually produce content. By default, treat the entire text as authored by a single speaker unless step 2 proved otherwise.
4. Decide whether the text is written ABOUT a person (third-person) or BY a person (first-person).
5. Check whether the lines are contiguous in meaning (a connected thesis or logical flow) or are discrete, disjointed standalone statements from the same speaker.
6. Identify whether a single individual is the clear subject or primary speaker; if yes, record their name in `target_name`, otherwise set `target_name` to null.
7. Select the single best-matching `classified_situation` based on the DOMINANT mode of the text. If `dialogue` was eliminated in step 2, choose between `tweets_or_quotes`, `monologue`, and `biographical_facts`.
8. Write clear, evidence-based reasoning that cites specific phrases or structural cues from the text. If you chose `dialogue`, your reasoning MUST quote the two or more explicit speaker labels and the statements directly attributed to each.
 
# Output Format

Return a structured object with the following fields:
- `classified_situation`: exactly one of `biographical_facts`, `dialogue`, `tweets_or_quotes`, `monologue`.
- `reasoning`: step-by-step reasoning that cites specific evidence from the text.
- `has_identifiable_target`: `true` when a single named individual is the clear subject or primary speaker, otherwise `false`.
- `target_name`: the full name (or best identifier) of the primary target individual if one can be identified; otherwise `null`. Use only names that appear explicitly in the text.

# Examples

## Example 1 — dialogue
"segments": [{"id": "seg_0", "end": 4.65, "speaker": "A", "start": 0.0, "text": " We've been working on a device that can read your mind and we would love to see your thoughts.", "type": "transcript.text.segment"}, {"id": "seg_1", "end": 8.850000000000001, "speaker": "avatar", "start": 8.000000000000002, "text": " Is that the joke?", "type": "transcript.text.segment"}]

## Example 2 — tweets_or_quotes
Always something new for the magazine cover and the articles practically write themselves

## Example 3 — tweets_or_quotes
@Tesmanian_com Zip around Vegas super fast with Teslas in tunnels!

## Example 4 — tweets_or_quotes
### Each of the following lines is a distinct, disjointed statement. There are NO explicit speaker labels, no `Name:` prefixes, and no JSON `speaker` fields. Different topics, different recipients, and different tones across lines do NOT indicate different speakers. Classify this as `tweets_or_quotes`, not `dialogue` (line-separated disjointed statements from ONE speaker — NOT a dialogue).

Ms Melissa just passed!!!! I bet she's bitching God out for taking her away from her family, about now!
Love you both .... I am fine
It's ok now ... everything is ok now ... she's no longer struggling and no longer in pain ....
@friend1 happy birthday!!
just landed in Tokyo, what a flight
why do my keys always disappear right when I need to leave
the new album is unreal, on repeat all morning
hope everyone is having a good Tuesday
ok back to work
Melissa just passed. Will you send flowers?

## Example 5 — biographical_facts
She is loved by her child.
She makes her child laugh.
She cooks food that her child thinks is the best.
She taught her child to work hard.
She fights aging with pomegranate shots.
She is kind.
She is playfully ornery!
She gives her child shelter.
She heals the sick.
She shows her child strength of character.
She is rich in soul.
She introduces her child to prayer.
She never backs down.
She is her child's hero, angel, and world.
She is 57 years old.

## Example 6 — monologue
There is true synergy which blooms from working smart in tandem with working hard. You will maximize your potential if you work smart then hard. If you work hard but not smart, you run the risk of wasting time & energy. You will need to fail to learn from your mistakes, and the faster you fail and recover, the faster you can learn from your mistakes and continue to grow. If you are working hard while not learning from your mistakes, you are wasting large portions of energy. On the contrary, if you work smart, but not hard, then you are limiting your self of your true potential, and may never realize success or what lies beyond perceived success, as a consequence. I have lived by the motto: "Things that are worth doing in life are not easy". It would be smart to choose work with which to live a worthwhile life.

# Context

- Your output is consumed by a downstream data pipeline whose routing logic depends entirely on the chosen `classified_situation`. A wrong classification sends the document through the wrong processing path.
- Inputs may be raw text, line-delimited statements, or JSON transcripts containing speaker-labeled segments.
- A common failure mode to AVOID: line-separated text from a single person (e.g. tweets, status updates, chat messages from one person, journal fragments) being mistakenly classified as `dialogue` because the lines look like different "responses". They are NOT different responses. Each line is a distinct, disjointed statement from the SAME speaker unless there are explicit speaker labels.
- The presence of names, @-mentions, hashtags, JSON structure, line breaks, topic shifts, tone shifts, or message-style formatting alone does NOT establish a back-and-forth conversation. Only an explicit speaker label attached to each turn establishes a dialogue.
- A single speaker mentioning multiple names is NOT a dialogue.
- A list of standalone, non-contiguous statements without explicit speaker labels is `tweets_or_quotes`, not `dialogue`.
- When in doubt, prefer the single-speaker (tweets_or_quotes, monologue, biographical_facts) classification.

# Final instructions and prompt to think step by step

Think step by step before answering.

1. SCAN the text for explicit speaker labels (inline `Name:` / `Name -` prefixes attached to a statement, screenplay-style label lines paired with a following utterance, or a structured `speaker`/`from`/`author` field paired with a `text` field in JSON). If you cannot point to TWO OR MORE distinct labeled entities each producing at least one labeled statement, ELIMINATE `dialogue` from consideration immediately.
2. Determine if the text is third-person (about a person) or first-person (by a person).
3. Check whether the lines form a connected, contiguous flow of meaning (likely `monologue`) or are discrete, disjointed standalone statements from the same speaker (likely `tweets_or_quotes`).
4. Identify the single primary target individual if one exists, using only names that appear explicitly in the text. Do not invent a `target_name`.
5. Select the single best-matching `classified_situation` based on the DOMINANT mode of the text. If `dialogue` was eliminated in step 1, choose between `tweets_or_quotes`, `monologue`, and `biographical_facts`.
6. Write reasoning that cites specific evidence from the text. If you chose `dialogue`, your reasoning MUST quote the two or more explicit speaker labels and the statements directly attributed to each — otherwise change your classification.

Remember: line breaks, name mentions, @-mentions, topic shifts, and tone shifts are NOT speaker changes. The default prior is a single speaker. Only explicit, structural speaker labels override that prior.
"""


# ===========================================================
# STEP 2b — Useful-Content vs Fragment Classification
# ============================================================


class UsefulContentClassification(BaseModel):
    """
    Decides whether a chunk of extracted text is meaningful identity / quote /
    biographical content, or boilerplate noise (page numbers, running headers /
    footers, navigation menus, timestamps, cookie banners, ads). Used as the
    LLM fallback for indeterminant chunks the cheap heuristic can't decide.
    """

    is_useful: bool = Field(
        description=(
            "True when the text carries real semantic content about a person, "
            "their words, or a topic. False when it is boilerplate / navigation "
            "/ page furniture with no standalone meaning."
        )
    )
    reasoning: str = Field(
        description="Brief justification citing what in the text drove the decision."
    )


USEFUL_CONTENT_CLASSIFICATION_SYSTEM_PROMPT = """
# Role and Objective

You are a data-quality gate for a document-ingestion pipeline. You receive a
single short chunk of text that was extracted from a PDF, a web page, or a
transcript. Decide whether the chunk is USEFUL content worth storing, or a
FRAGMENT of boilerplate that should be discarded.

# What counts as USEFUL (is_useful = true)
- A sentence or statement that conveys information, opinion, narrative, or dialogue.
- A direct quote, a biographical fact, a list item with real meaning, a heading
  that labels real content followed by substance.
- Even a short line is useful if it is a genuine standalone statement
  (e.g. a tweet, an aphorism, a spoken line).

# What counts as a FRAGMENT (is_useful = false)
- Page numbers and pagination ("Page 13 of 10", "12", "- 4 -").
- Running headers / footers repeated across pages.
- Date/time stamps and print artifacts ("11/5/25, 11:46 PM").
- Navigation menus, breadcrumbs, "Skip to content", "Share", "Subscribe".
- Cookie / consent banners, copyright lines, ad slugs, "Read more".
- Pure punctuation, separators, or whitespace.
- Strings with no parseable meaning out of context.

# Reasoning Steps
1. Read the chunk.
2. Ask: if a human read ONLY this chunk, would it tell them something real about
   a person, their words, or a subject?
3. If yes -> is_useful = true. If it is page furniture / navigation / noise ->
   is_useful = false.
4. When genuinely uncertain, lean toward is_useful = true (favor recall; the
   cheap heuristic already removed the obvious garbage).

# Output Format
Return a structured object with `is_useful` (bool) and `reasoning` (str).
"""


class TitleFragmentClassification(BaseModel):
    """
    Decides whether a single extracted line is a restatement of the document's
    title — i.e. page furniture (a running header / nav bar / cover title) rather
    than body content. Used as the LLM safeguard for the fragment filter when the
    cheap embedding-similarity check between the line and the page title is in the
    ambiguous band (high enough to be suspicious, below the auto-drop threshold).
    Embedding similarity alone misclassifies lines that share a repeated brand or
    keyword with the title (e.g. a paragraph that says "iPhone" several times),
    so this judge looks at meaning and role rather than lexical overlap.
    """

    is_title: bool = Field(
        description=(
            "True when the line is the document/page title or a running "
            "header/footer restating it — page furniture with no standalone "
            "body meaning. False when it is real body content that merely shares "
            "words with the title."
        )
    )
    reasoning: str = Field(
        description="Brief justification citing what in the line drove the decision."
    )


TITLE_FRAGMENT_CLASSIFICATION_SYSTEM_PROMPT = """
# Role and Objective

You are a data-quality gate for a document-ingestion pipeline. You receive the
document's TITLE and a single LINE extracted from one of its pages. Decide whether
the LINE is just the title repeated as page furniture (a cover title, a running
header/footer, or a nav bar restating the title), or whether it is real body
content that happens to share words with the title.

# Why you exist

A cheap embedding-similarity check already runs before you. It drops lines that
are almost identical to the title and keeps lines that are clearly unrelated. You
are only called for the ambiguous middle: lines that overlap the title enough to
be suspicious. The known failure mode of similarity alone is lexical repetition —
a paragraph that names a brand or keyword from the title several times (e.g.
"iPhone ... iPhone ... iPhone") scores as title-like even though it is genuine
content. Judge by MEANING and ROLE, not by shared words.

# Mark is_title = true when
- The line is the title (or an obvious truncation/variant of it) standing alone.
- The line is a running header/footer or nav element that restates the title on
  every page.
- The line carries no standalone statement — it only names the document/topic.

# Mark is_title = false when
- The line is a sentence, claim, quote, or narrative that conveys information,
  even if it repeats a word from the title.
- The line is a genuine heading that introduces real content distinct from the
  title.
- When genuinely uncertain, lean toward is_title = false (favor keeping real
  content; the similarity gate already removed near-identical title copies).

# Output Format
Return a structured object with `is_title` (bool) and `reasoning` (str).
"""


DESCRIBE_IMAGE_PROMPT = """
<describe_image_spec>
<role>
You are a vision analyst. Your outputs stand in for the pixels so downstream
systems can search, summarize, and reason about the scene. Optimize for
faithfulness and retrieval coverage over creative interpretation.
</role>

<task>
Describe exactly one user-supplied image per request. The image is the only
evidence; do not assume off-image context unless the image itself supplies it
(e.g. captions, UI chrome).
</task>

<instruction_hierarchy>
1. Fidelity first: do not invent objects, people, readable text, or actions not
   supported by what is visible. When detail is unclear, say so briefly instead
   of guessing.
2. Coverage second: address layout, salient subjects, clothing and pose,
   expressions and actions when visible, environment and lighting, palette,
   visible text (transcribe short strings literally), logos, UI elements, and
   spatial relationships that matter for understanding the scene.
3. Tone: neutral third person. Do not role-play as anyone shown in the image. Make no mention that this is an image.
4. Single-turn completion: deliver the full description in one reply. Do not ask
   clarifying questions, do not request another image, and do not defer.
</instruction_hierarchy>

<output_contract>
- Open with 3–5 very short bullet lines (one fragment per line) that anchor the
  main subjects, setting, and any critical visible text or UI. This orients
  readers before the narrative detail.
- Follow with one to three paragraphs of continuous prose that elaborate for
  semantic search—relationships, atmosphere, and specifics not already repeated
  verbatim in every bullet.
- Do not emit JSON, YAML, or other machine-oriented schemas.
- Do not use meta labels such as "Description:" or "Here is my analysis:".
- Use Markdown only where it clearly helps (e.g. a short `inline` label); default
  to plain sentences and line breaks. Do not build the answer from headings alone.
- Make no mention that this is an image.
- Do not preface the description with description. 
- ONLY include the description. 
- NEVER include a preface proceeding the description.
</output_contract>

<escape_hatches>
- Blank, nearly blank, or non-representational input: state that in the bullets
  and one short paragraph; still honor the contract.
- Genuine ambiguity: choose the most plausible reading, note the ambiguity once
  in prose, and proceed—do not stall or hand back to the user.
</escape_hatches>

<scope_control>
Prefer a correct moderate-length answer over exhaustive micro-description of
every background pixel. Skip trivial clutter unless it changes meaning.
</scope_control>

<anti_patterns>
Avoid internally conflicting goals (for example demanding both pixel-perfect
inventory and extreme brevity). If instructions appear to collide, follow
fidelity and the instruction_hierarchy order above.
Make no mention that this is an image of any type.
Do not preface the description with description. 
Example (DO NOT DO THIS): 'I can’t write in the first person as if I were the person in the photo, but here is a vivid, third-person portrayal of her:
</anti_patterns>
</describe_image_spec>
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
- Do NOT exclude brackets or narrative markers from extracted text.
- Copy the text completely and do not alter the text in any way.
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
    content: str = Field(
        description="Verbatim or near-verbatim content of the statement."
    )
    is_target: bool = Field(
        description="True if this speaker is the identified target individual."
    )


class TargetIdentificationInText(BaseModel):
    """
    Given conversational text, identifies the TARGET individual and labels
    every speaker turn. This is used between Step 2 and Step 3.
    """

    target_name: str = Field(
        description="Full name (or best identifier) of the target individual."
    )
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
# STEP 5a-bis — Diarization Speaker Reconciliation
# ============================================================


class ReconciledSpeakerLabel(BaseModel):
    raw_speaker: str = Field(
        description="A speaker label exactly as produced by the diarizer (e.g. 'A', 'B', 'C')."
    )
    canonical_speaker: str = Field(
        description=(
            "The corrected speaker this raw label really belongs to. Raw labels "
            "that are the same underlying person share one canonical_speaker. The "
            "target individual's canonical_speaker MUST be the literal string 'avatar'."
        )
    )
    is_target: bool = Field(
        description="True only for the canonical speaker that is the target individual."
    )


class SpeakerReconciliation(BaseModel):
    """
    Corrects over/under-splitting from ``gpt-4o-transcribe-diarize``. The diarizer
    has no parameter to set the number of speakers and frequently splits one
    person across several labels (or, rarely, merges two). Given the labeled,
    coalesced turns and the known target name, this produces a mapping from each
    raw diarizer label to a corrected canonical speaker, and identifies which
    canonical speaker is the target (relabeled 'avatar').
    """

    label_map: List[ReconciledSpeakerLabel] = Field(
        description="One entry per DISTINCT raw diarizer label, mapping it to its canonical speaker."
    )
    canonical_speaker_count: int = Field(
        description="The corrected number of distinct real speakers in the conversation."
    )
    reasoning: str = Field(
        description="Concise evidence (voice continuity, turn-taking, content cues) for the merges/splits."
    )


SPEAKER_RECONCILIATION_SYSTEM_PROMPT = """
# Role and Objective

You correct speaker-diarization labels for a transcript. An automatic diarizer
labeled each turn with a speaker tag (e.g. "A", "B", "C"), but it often SPLITS a
single real person across multiple tags, and occasionally MERGES two people under
one tag. Using the conversation content and turn-taking, produce a corrected
mapping from each raw label to a canonical speaker.

# Inputs
- The known TARGET speaker label/name (the persona being reconstructed).
- The list of turns, each with its raw diarizer speaker tag and text.

# Instructions
1. Read the whole transcript and follow the flow of who is speaking.
2. Group raw labels that are clearly the SAME person (consistent voice, role,
   self-reference, and coherent continuation across turns) under one
   `canonical_speaker` name.
3. Keep genuinely distinct people as distinct canonical speakers.
4. The TARGET individual's `canonical_speaker` MUST be exactly the string
   "avatar", and its `is_target` MUST be true. All other canonical speakers have
   `is_target` = false.
5. Output exactly one `label_map` entry per DISTINCT raw label that appears.
6. Set `canonical_speaker_count` to the number of distinct canonical speakers.

# Guidance
- An interviewer/host who asks the questions is usually ONE person even if the
  diarizer split them into several tags between the target's long answers.
- Prefer FEWER canonical speakers when evidence is ambiguous — over-splitting is
  the diarizer's common failure mode.
- Do not invent speakers that have no turns.

# Output Format
Return `label_map` (one entry per distinct raw label), `canonical_speaker_count`
(int), and `reasoning` (str).
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
    value: str = Field(
        description="Full name or best available identifier of the target individual."
    )
    evidence: str = Field(
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
    """Extracts Description of a Target individual."""

    description: str = Field(
        description="A concise one-to-two paragraph summary of who this person is."
    )

    reasoning: str = Field(
        description="How the summary was constructed from the available text."
    )

    evidence: str = Field(
        description="Key passages that were synthesised to form this description."
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

Describe the following target individual: 

{target_individual}

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
        description=(
            "How the person defines themselves: profession, role, community, "
            "cultural identity, or other primary self-concept."
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Key biographical facts, background, and formative events, "
            "presented in roughly chronological order."
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Dominant or recurring emotional states expressed or implied "
            "by the target. Each item is a concise label + brief explanation, "
            "e.g. 'Persistent optimism — regularly frames setbacks as growth.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Core beliefs or worldview statements. Each item is a declarative "
            "sentence capturing one belief, e.g. 'People are fundamentally good.'"
        )
    )
    evidence: Optional[str] = Field(description="Passages that reveal these beliefs.")
    confidence: ConfidenceLevel = Field(
        description="Confidence in the belief characterisation."
    )
    reasoning: str = Field(description="How each belief was inferred from the text.")


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
        description=(
            "What the person demonstrably prioritises and protects. These are synonymous with priorities. "
            "Each item names the value and briefly explains how it is demonstrated, "
            "e.g. 'Family — consistently prioritises family time over career advancement.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Specific stated or strongly implied opinions on concrete topics. "
            "Each item includes the topic and the opinion, "
            "e.g. 'On social media: believes it damages genuine human connection.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Long-term ambitions and objectives. Each item is a concise statement "
            "of a goal, e.g. 'Build a company that outlasts its founder.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Immediate desires or things the person is actively pursuing right now. "
            "e.g. 'Wants public recognition for recent work.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Deeper psychological or practical needs, stated or implied. "
            "e.g. 'Needs external validation to feel secure in decisions.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Things the person is afraid of or actively avoids. "
            "e.g. 'Fear of irrelevance — avoids topics where their authority could be challenged.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Acknowledged or observable weaknesses, blind spots, or contradictions. "
            "e.g. 'Impatience — frequently interrupts others and rushes decisions.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Demonstrated strengths, skills, or exceptional qualities. "
            "e.g. 'Strategic clarity — consistently identifies the core issue quickly.'"
        )
    )
    evidence: Optional[str] = Field(
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
        description=(
            "Active problems, challenges, conflicts, or obstacles the target "
            "is facing or has faced. "
            "e.g. 'Strained relationship with co-founder creating organisational tension.'"
        )
    )
    evidence: Optional[str] = Field(description="Passages that reveal these problems.")
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
        description="Named individuals and the nature of their relationship to the target."
    )
    evidence: Optional[str] = Field(
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
# Master registry — maps dimension name → (model class, system prompt)
# Canonical order matches GeneralCharacteristicExtraction field order.
# Useful for iterating all analyses programmatically.
# ─────────────────────────────────────────────────────────────

CHARACTERISTIC_EXTRACTORS: dict[str, tuple[type[BaseModel], str]] = {
    "name": (NameExtraction, NAME_EXTRACTION_SYSTEM_PROMPT),
    "description": (DescriptionExtraction, DESCRIPTION_EXTRACTION_SYSTEM_PROMPT),
    "identity": (IdentityExtraction, IDENTITY_EXTRACTION_SYSTEM_PROMPT),
    "history": (HistoryExtraction, HISTORY_EXTRACTION_SYSTEM_PROMPT),
    "emotions": (EmotionsExtraction, EMOTIONS_EXTRACTION_SYSTEM_PROMPT),
    "beliefs": (BeliefsExtraction, BELIEFS_EXTRACTION_SYSTEM_PROMPT),
    "values": (ValuesExtraction, VALUES_EXTRACTION_SYSTEM_PROMPT),
    "opinions": (OpinionsExtraction, OPINIONS_EXTRACTION_SYSTEM_PROMPT),
    "goals": (GoalsExtraction, GOALS_EXTRACTION_SYSTEM_PROMPT),
    "wants": (WantsExtraction, WANTS_EXTRACTION_SYSTEM_PROMPT),
    "needs": (NeedsExtraction, NEEDS_EXTRACTION_SYSTEM_PROMPT),
    "fears": (FearsExtraction, FEARS_EXTRACTION_SYSTEM_PROMPT),
    "problems": (ProblemsExtraction, PROBLEMS_EXTRACTION_SYSTEM_PROMPT),
    "flaws": (FlawsExtraction, FLAWS_EXTRACTION_SYSTEM_PROMPT),
    "strengths": (StrengthsExtraction, STRENGTHS_EXTRACTION_SYSTEM_PROMPT),
    "relationships": (RelationshipsExtraction, RELATIONSHIPS_EXTRACTION_SYSTEM_PROMPT),
}


class Relationship(BaseModel):
    person: str
    relationship_type: str = Field(
        description="e.g. 'close friend', 'rival', 'mentor', 'romantic partner'"
    )
    description: str = Field(
        description="Brief description of the dynamic between the target and this person."
    )


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

    name: Optional[str] = Field(description="Full name of the target individual.")
    description: Optional[str] = Field(
        description="One-paragraph summary of who this person is."
    )
    identity: Optional[str] = Field(
        description="How the person identifies themselves (profession, role, community, etc.)."
    )
    history: Optional[str] = Field(
        description="Key biographical facts, background, formative events."
    )
    emotions: Optional[List[str]] = Field(
        description="Dominant or recurring emotional states expressed or implied."
    )
    beliefs: Optional[List[str]] = Field(
        description="Core beliefs or worldview statements inferred from the text."
    )
    values: Optional[List[str]] = Field(
        description="What the person demonstrably cares about most."
    )
    opinions: Optional[List[str]] = Field(
        description="Specific stated or strongly implied opinions on topics."
    )
    goals: Optional[List[str]] = Field(description="Long-term ambitions or objectives.")
    wants: Optional[List[str]] = Field(
        description="Immediate desires or things the person is actively pursuing."
    )
    needs: Optional[List[str]] = Field(
        description="Deeper psychological or practical needs, stated or implied."
    )
    fears: Optional[List[str]] = Field(
        description="Things the person is afraid of or actively avoids."
    )
    problems: Optional[List[str]] = Field(
        description=(
            "Current challenges the individual is facing. These could be internal "
            "conflicts or external conflicts that the individual needs to resolve."
        )
    )
    flaws: Optional[List[str]] = Field(
        description="Acknowledged or observable weaknesses, blind spots, or contradictions."
    )
    strengths: Optional[List[str]] = Field(
        description=(
            "Features of the individual that the individual excels at. The best "
            "qualities of the individual and what the individual is best at doing "
            "or what is best about the individual."
        )
    )
    relationships: Optional[List[RelationshipEntry]] = Field(
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
- Always populate the reasoning field with your full chain of thought,
  citing the specific textual evidence for each populated dimension.
</Rules>
"""


# ============================================================
# CSV column identification (used for tabular ingest preprocessing)
# ============================================================


class CSVUserTextColumnIdentification(BaseModel):
    """Identify which CSV column carries free-text from the target speaker, and
    where their name comes from.

    The model is constrained to choose ``text_column`` strictly from the
    supplied header list. ``target_name_column`` is optional because some CSVs
    have a single, implicit target (e.g. ``elon_musk_tweets.csv``) so the name
    must instead be inferred from the dominant column value or the filename;
    ``target_name_value`` is filled in by Python after the call when the data
    itself is the source of truth.
    """

    text_column: str = Field(
        description=(
            "Name of the single column whose cells contain the verbatim "
            "free-text statement written or spoken by the target individual "
            "(the post body, tweet text, message body, transcript line, "
            "etc.). Must match one of the supplied header names exactly."
        )
    )
    target_name_column: Optional[str] = Field(
        default=None,
        description=(
            "Name of the column whose cells hold the target individual's "
            "name (or username/handle) when one exists. Null when no column "
            "is a clear name source — e.g. when every row is implicitly the "
            "same person and no name field is present."
        ),
    )
    target_name_value: Optional[str] = Field(
        default=None,
        description=(
            "Best-effort target name when the same name dominates the data "
            "(e.g. every row has user_name='Elon Musk') OR when a name is "
            "obvious from the supplied filename. Null if neither source is "
            "trustworthy. Must be a name of a person, not a topic."
        ),
    )
    reasoning: str = Field(
        description=(
            "Brief justification — which header was chosen for text_column, "
            "why the target column or target_name_value was picked, and any "
            "rejected alternatives."
        )
    )


CSV_USER_TEXT_COLUMN_IDENTIFICATION_SYSTEM_PROMPT = """<role>
You are a tabular data analyst preparing a CSV upload for ingestion into an
avatar identity store. Your job is to identify (a) which column contains the
verbatim free-text statements written or spoken by the target individual, and
(b) where the target's name comes from — either a column whose cells hold the
name, a single dominant value across the data, or a hint from the filename.
</role>

<task>
You will receive:
  * The CSV filename.
  * The full ordered list of column headers.
  * A small sample of rows as structured records, plus per-column summary
    stats (sample non-empty values, distinct-value count, average cell
    length).

Return a single ``CSVUserTextColumnIdentification`` object:
  - text_column: must match one of the supplied headers exactly.
  - target_name_column: a header from the list, or null.
  - target_name_value: a literal person/handle string when the data or
    filename make one obvious; otherwise null.
  - reasoning: short, evidence-based.
</task>

<instruction_hierarchy>
1. text_column must be the column with long, free-form, human-authored text
   (posts, tweets, transcript turns, messages). It is almost never a numeric
   id, timestamp, URL, hashtag list, source/device label, count, or boolean.
2. target_name_column is a column whose cells carry the speaker's display
   name, username, screen name, handle, author, or speaker label. Common
   header names: ``user_name``, ``username``, ``user``, ``name``, ``author``,
   ``screen_name``, ``handle``, ``creator``, ``speaker``, ``full_name``.
3. target_name_value should be filled when:
     a. one specific person/handle dominates the target_name_column (>=80% of
        rows share the same value) — return that value, or
     b. the filename itself clearly names a single person (e.g.
        ``elon_musk_tweets.csv`` -> "Elon Musk").
   Otherwise return null and let the per-row column value carry the name.
4. Prefer null over guessing. If no header is a believable text column,
   choose the closest free-text column anyway (text_column is required) but
   say so in reasoning.
</instruction_hierarchy>

<anti_patterns>
- Do NOT pick a numeric, boolean, date, URL, or hashtag column as
  text_column.
- Do NOT invent a column name that is not in the supplied headers.
- Do NOT set target_name_value to a topic, brand, location, or job title —
  it must be a person's name or handle.
- Do NOT echo the sample rows back; only return the structured fields.
</anti_patterns>

<output_format>
Return only the structured CSVUserTextColumnIdentification object the runtime
expects. No prose outside the structured fields.
</output_format>
"""
