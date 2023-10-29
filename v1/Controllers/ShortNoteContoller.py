from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse

import os
import PyPDF2
import re
import math
from typing import List

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googletrans import Translator
from gtts import gTTS


import v1.Services.ShortNote.ShortNote as ShortNote

UPLOAD_FOLDER = "uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

STATIC_DIR = "C:/Users/Sayanthan/Documents/RP_Project/rp_forntend/src/genrateImage"


def count_word_range(file: UploadFile):
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    text = extract_text_from_pdf(filepath)
    word_count = len(text.split())
    min_count = 100
    max_count = int(word_count * 0.50)

    print("======")
    print(word_count)

    return {
        "minCount": min_count,
        "maxCount": max_count,
        "filename": file.filename,
        "word_count": word_count,
    }


def get_summary(filename: str, summaryLength: int):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail=f"File {filepath} does not exist")

    text = extract_text_from_pdf(filepath)
    print("text\n")
    print(text)

    word_chunk_count = len(text.split()) / 450
    number_of_chunks = math.ceil(word_chunk_count)
    print("wordcount")
    print(number_of_chunks)
    print(len(text.split()))
    max_gen_length_model = int(1.5 * summaryLength / number_of_chunks)
    min_gen_length_model = int(1.2 * summaryLength / number_of_chunks)
    print("max\n")
    print(max_gen_length_model)
    print("min\n")
    print(min_gen_length_model)

    word_count = len(text.split())
    min_count = 100
    max_count = int(word_count * 0.50)

    # Check if the provided summaryLength is within the valid range
    if not min_count <= summaryLength <= max_count:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid summaryLength. It should be between {min_count} and {max_count} words.",
        )

    chunks = chunk_text(text)
    print("chunks\n")
    print(chunks)

    final_summary = ShortNote.generate_summary_using_model(
        chunks, max_gen_length_model, min_gen_length_model
    )

    if final_summary is not None:
        img_path = generate_wordcloud(final_summary)
        final_summary_now = remove_duplicate_sentences(final_summary)

        return {"summary": final_summary_now, "wordcloud_image": img_path}


def get_translation(language: str, text: str):
    translator = Translator()
    language = language
    text = text
    if language not in ["si", "ta"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    translated_text = translator.translate(text, src="en", dest=language).text
    return {"translatedText": translated_text}


def get_audio(context):
    # Create a gTTS object
    tts = gTTS(context)

    # Save the audio file
    tts.save("shortnote_audio.mp3")

    # Provide a download link
    return FileResponse(
        "shortnote_audio.mp3",
        headers={"Content-Disposition": "attachment; filename=shortnote_audio.mp3"},
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        extracted_text = "".join(page.extract_text() for page in pdf_reader.pages)
    # Cleaning up the extracted text
    cleaned_text = re.sub(r"[-•●]", "", extracted_text)
    cleaned_text = re.sub(r"\b[A-Z\s]{2,}\b", "", cleaned_text)
    cleaned_text = re.sub(r"\b[A-Za-z\s]+:\b", "", cleaned_text)
    return " ".join(cleaned_text.split())


def chunk_text(text: str, n=450) -> List[str]:
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(0, len(words), n)]


def generate_wordcloud(text: str) -> str:
    wordcloud = WordCloud(
        background_color="white", width=800, height=800, max_words=100
    ).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    img_path = os.path.join(STATIC_DIR, "wordcloud.png")
    plt.savefig(img_path, format="png")
    return img_path


def remove_duplicate_sentences(text, min_appearances=2):
    sentences = text.split(". ")
    sentence_frequencies = {}
    to_remove = set()

    for i, sentence in enumerate(sentences):
        if sentence in sentence_frequencies:
            sentence_frequencies[sentence] += 1
        else:
            sentence_frequencies[sentence] = 1

        if sentence_frequencies[sentence] >= min_appearances:
            to_remove.add(i)

    return ". ".join(
        [sentences[i] for i in range(len(sentences)) if i not in to_remove]
    )
