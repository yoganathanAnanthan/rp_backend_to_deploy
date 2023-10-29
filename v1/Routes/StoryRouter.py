from fastapi import APIRouter, Form

import v1.Controllers.StoryController as StoryController


story_router = APIRouter()


@story_router.post("/")
def create_story(
    genre: str = Form(...),
    prompt: str = Form(...),
):
    return StoryController.make_story(genre, prompt)
    # return {
    #     "story": "Once upon a time there was an old mother pig who had three little pigs and not enough food to feed them. So when they were old enough, she sent them out into the world to seek their fortunes"
    # }


@story_router.post("/video")
def create_story(
    story: str = Form(...),
):
    return StoryController.make_video(story)
    # return {"video": "Video File we will send"}
