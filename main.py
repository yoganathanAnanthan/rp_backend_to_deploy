from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from v1.Routes.MCQRouter import mcq_router
# from v1.Routes.EssayRouter import essay_router
# from v1.Routes.ShortNoteRouter import short_note_router
# from v1.Routes.StoryRouter import story_router

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mcq_router, prefix="")
# app.include_router(essay_router, prefix="/essay")
# app.include_router(short_note_router, prefix="/short_note")
# app.include_router(story_router, prefix="/story")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
