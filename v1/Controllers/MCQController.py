from fastapi.responses import FileResponse
from gtts import gTTS

# MCQs Component APIs
from v1.Services.MCQs.Summarizer import summarizer
from v1.Services.MCQs.KeywordExtraction import get_keywords
from v1.Services.MCQs.FilterBestKeywords import filter_keywords
from v1.Services.MCQs.QuestionGeneration import get_question
from v1.Services.MCQs.DistractorsGeneration import generate_distractors


def generate_qanda_pairs(context, number):
    summarized_text = summarizer(context)
    imp_keywords = get_keywords(context, summarized_text)
    imp_fil_keywords = filter_keywords(imp_keywords, num_best_keywords=number)

    qa_pairs = []

    for answer in imp_fil_keywords:
        ques = get_question(summarized_text, answer)
        qa_pairs.append({"question": ques, "answer": answer.capitalize()})

    return qa_pairs


def generate_mcqs(context, number):
    summarized_text = summarizer(context)
    imp_keywords = get_keywords(context, summarized_text)
    imp_fil_keywords = filter_keywords(imp_keywords, num_best_keywords=number)

    mcq = []

    for answer in imp_fil_keywords:
        ques = get_question(summarized_text, answer)
        distractors = generate_distractors(ques, answer)
        mcq.append(
            {
                "question": ques,
                "answer": answer.capitalize(),
                "distractors": distractors,
            }
        )

    return mcq


def get_audio(context):
    # Create a gTTS object
    tts = gTTS(context)

    # Save the audio file
    tts.save("output.mp3")

    # Provide a download link
    return FileResponse(
        "output.mp3", headers={"Content-Disposition": "attachment; filename=output.mp3"}
    )
