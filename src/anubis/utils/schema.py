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

