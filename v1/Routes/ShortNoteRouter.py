from fastapi import APIRouter
from fastapi import FastAPI, UploadFile, HTTPException

import v1.Controllers.ShortNoteContoller as ShortNoteContoller
from v1.Models.ShortNoteModel import (
    WordCountResponse,
    SummaryRequest,
    TranslationRequest,
    GenerateAudioData,
)

short_note_router = APIRouter()


@short_note_router.post("/word_count_range", response_model=WordCountResponse)
async def get_word_count_range(file: UploadFile = UploadFile(...)):
    return ShortNoteContoller.count_word_range(file)


@short_note_router.post("/summarize1")
async def summarize(summary_request: SummaryRequest):
    return ShortNoteContoller.get_summary(
        summary_request.filename, summary_request.summaryLength
    )


@short_note_router.post("/translate")
async def translate(translation_request: TranslationRequest):
    return ShortNoteContoller.get_translation(
        translation_request.language, translation_request.text
    )


@short_note_router.post("/generate_audio")
async def generate_audio(input_data: GenerateAudioData):
    return ShortNoteContoller.get_audio(input_data.text)
