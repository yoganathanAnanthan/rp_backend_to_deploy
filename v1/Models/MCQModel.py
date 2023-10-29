from pydantic import BaseModel

class InputData(BaseModel):
    context: str
    number: int