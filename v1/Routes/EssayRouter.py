from fastapi import APIRouter, Form
from v1.Models.EssayModel import EssayRequest
import v1.Controllers.EssayController as EssayController


essay_router = APIRouter()


@essay_router.post("/")
def get_socre(request: EssayRequest):
    return EssayController.calculate_essay_score_v2(request.topic, request.essay)


# @essay_router.post("/")
# def get_socre(topic: str = Form(...), essay: str = Form(...)):
#     return EssayController.calculate_essay_score(topic, essay)
