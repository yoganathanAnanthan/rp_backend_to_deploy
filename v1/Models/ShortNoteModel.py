from pydantic import BaseModel
from typing import List, Optional, TypeVar, Generic

T = TypeVar("T")


class WordCountResponse(BaseModel):
    minCount: int
    maxCount: int
    filename: str
    word_count: int


class SummaryRequest(BaseModel):
    filename: str
    summaryLength: int


class TranslationRequest(BaseModel):
    language: str
    text: str


class GenerateAudioData(BaseModel):
    text: str
