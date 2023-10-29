from pydantic import BaseModel


class EssayRequest(BaseModel):
    topic: str
    essay: str
