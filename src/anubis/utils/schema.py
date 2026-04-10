# src/anubis/utils/classes

from pydantic import BaseModel, Field
from typing import Literal

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

class MonologuePresentationOrSeriesOfQuotes(BaseModel):          
    """ Determine if this is a fluid train of thought as if in a monologue or presentation for a single speaker or if this is a series of direct quotes from the speaker. """
    classified_situation: Literal["MonologueOrPresentation", "SeriesOfDistinctQuotes"]
    reason: str = Field()

class ProprietaryContentClassification(BaseModel):
           """ Determine if the content is non-personally identifiable such as in a menu or a well known religious text such as the bible or is personally identifiable information such as in an interview, a monoluge, a persentation, or a script. """
           non_personally_identifiable_information: bool = Field(description= "This is TRUE when the text is a menu or well-known religious text such as the bible. This is FALSE otherwise and is intended to be FALSE when there is a script or a monologue or an interview between two people.")
           reasoning: str = Field(
               description = "Step-by-step reasoning behind the decision for the classified situation of the text. (Non-proprietary content includes: single speaker monologue, single tweet from user, strictly Q & A, multi-speaker that is Non-religious. There is a target to identify in the non-religious text document in non-proprietary content.)"
           )


""" ASSOCIATED PROMPTS """

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


