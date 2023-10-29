from fastapi import APIRouter

from v1.Models.MCQModel import InputData

import v1.Controllers.MCQController as MCQController

from typing import List

mcq_router = APIRouter()


@mcq_router.post("/qanda/", response_model=List[dict])
def get_qanda_pairs(input_data: InputData):
    context = input_data.context
    number = input_data.number
    return MCQController.generate_qanda_pairs(context, number)


@mcq_router.post("/mcqs/", response_model=List[dict])
def get_mcqs(input_data: InputData):
    context = input_data.context
    number = input_data.number
    return MCQController.generate_mcqs(context, number)


@mcq_router.post("/generate_audio")
async def generate_audio(input_data: InputData):
    context = input_data.context
    return MCQController.get_audio(context)
