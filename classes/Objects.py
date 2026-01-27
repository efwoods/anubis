from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated

class HealthResponse(BaseModel):
    status: str


class AvatarContext(TypedDict):
    name: str
    description: str
    facts: list[str]