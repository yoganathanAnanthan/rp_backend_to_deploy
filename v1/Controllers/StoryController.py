import v1.Services.Story.Video_Creation.video_creator as Video
import v1.Services.Story.Story as Story
from v1.Services.Story.Video_Creation.services import translate_english_to_tamil
from fastapi.responses import FileResponse
from fastapi import HTTPException
import nltk

import re


def clean_story_text(generated_story):
    try:
        # Tokenize the generated story into sentences & Remove incomplete sentences and extra spaces
        sentences = nltk.sent_tokenize(generated_story)
        text = " ".join(
            [sentence.strip() for sentence in sentences if len(sentence) > 1]
        )

        valid_characters_pattern = re.compile(r"[\x20-\x7E]+")  # ASCII characters
        cleaned_text = "".join(valid_characters_pattern.findall(text))

        return cleaned_text
    except Exception as e:
        print("Error in Cleaning Story")
        print(e)
        return False


def make_story(genre: str, prompt: str):
    # Based on genere select story
    if genre == "general":
        model_name = "v1\Services\Story\TrainedModels\m9"
    elif genre == "mystery":
        model_name = "v1\Services\Story\TrainedModels\m10"
    elif genre == "horror":
        model_name = "v1\Services\Story\TrainedModels\m11"
    else:
        print("Genre is not specifed")
        return HTTPException(status_code=404, detail="Please select valid genre type")

    if not prompt:
        print("Prompt is not specifed")
        return HTTPException(status_code=404, detail="Prompt is not specifed")

    # Generate Story
    generated_story = Story.generate_story(prompt, model_name)

    cleaned_story = clean_story_text(generated_story)
    print("\nCleaned Story:\n")
    print(cleaned_story)

    if cleaned_story:
        return {"story": cleaned_story}
    elif generated_story:
        return {"story": generated_story}
    else:
        return {"code": "500", "error": "Error in Generating Story"}


def make_video(story: str):
    if story:
        # Create Video
        video = Video.make_story_as_video_v2(story)
        if video:
            # return {"video": video}
            return FileResponse(
                video,
                headers={f"Content-Disposition": "attachment; filename={video}"},
            )
        else:
            return {"code": "500", "error": "Error in Creating Story"}
    else:
        print("story is not specifed")
        return HTTPException(status_code=404, detail="Story is not given")
