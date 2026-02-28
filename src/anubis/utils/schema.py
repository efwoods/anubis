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

