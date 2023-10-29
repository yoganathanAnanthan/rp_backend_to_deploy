from gtts import gTTS
import re
from googletrans import Translator
from PIL import Image, ImageOps
import os
import requests

# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
import requests
from PIL import Image
from io import BytesIO
import shutil


from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.VideoClip import TextClip
from moviepy.config import change_settings

import cv2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag

# Download the NLTK stopwords dataset if you haven't already
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")


from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("UNSPLASH_API_KEY")
# API_KEY = os.getenv("UNSPLASH_API_KEY_II")
TAMIL_FONT_PATH = os.getenv("TAMIL_FONT_PATH")
IMAGEMAGICK_BINARY_PATH = os.getenv("IMAGEMAGICK_BINARY_PATH")

# Update the path to your ImageMagick executable
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY_PATH})


#######################################################################################################
#######################################################################################################
#######################################################################################################


def split_sentences(text):
    try:
        # Split the text into sentences using regular expressions
        sentences = re.split(r"(?<=[.!?]) +", text)
        # Remove empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences

    except Exception as e:
        print("########### Error in split_sentences function ###########")
        print(e)


def text_to_audio(text, language, output_file):
    try:
        tts = gTTS(text=text, lang=language)
        tts.save(output_file)
        return output_file
    except Exception as e:
        print("###########  Error in text_to_audio function ###########")
        print(e)


def translate_english_to_tamil(text):
    try:
        translator = Translator()
        translated = translator.translate(text, src="en", dest="ta")
        return translated.text
    except Exception as e:
        print("########### Error in translate_english_to_tamil function ##########")
        print(e)


def download_unsplash_image(query, output_file_path):
    try:
        # URL for searching Unsplash
        url = (
            f"https://api.unsplash.com/search/photos/?query={query}&client_id={API_KEY}"
        )

        response = requests.get(url)
        data = response.json()

        # Extract the first image URL and download
        if data["results"]:
            first_result = data["results"][0]
            image_url = first_result["urls"]["regular"]
            response = requests.get(image_url)

            img = Image.open(BytesIO(response.content))
            img_path = output_file_path
            img.save(img_path)  # Save the image
            # img.show()  # Display the image
            return img_path

        else:
            print("No images results found. Getting a random image.")
            image_url = "https://images.unsplash.com/photo-1475483768296-6163e08872a1?crop=entropy\u0026cs=tinysrgb\u0026fit=max\u0026fm=jpg\u0026ixid=M3w0OTQyNDR8MHwxfHNlYXJjaHwyfHxzdG9yeXRlbGxpbmd8ZW58MHx8fHwxNjk4NTUzMTQ0fDA\u0026ixlib=rb-4.0.3\u0026q=80\u0026w=1080"
            response = requests.get(image_url)

            img = Image.open(BytesIO(response.content))
            img_path = output_file_path
            img.save(img_path)  # Save the image
            return img_path

    except Exception as e:
        print("########### Error in download_unsplash_image function ##########")
        print(e)


def resize_images(images_path_list):
    try:
        # Define the target aspect ratio
        target_width = 1280
        target_height = 720
        i = 1
        # Loop through each image path
        resized_image_paths = []
        for image_path in images_path_list:
            # Open the image
            img = Image.open(image_path)

            # Calculate the current aspect ratio
            current_width, current_height = img.size
            current_ratio = current_width / current_height

            # Calculate the required dimensions to fit the target ratio
            if current_ratio > target_width / target_height:
                new_width = int(target_height * current_ratio)
                new_height = target_height
            else:
                new_width = target_width
                new_height = int(target_width / current_ratio)

            # Resize the image while maintaining its content and adding white background
            resized_img = ImageOps.pad(
                img.resize((new_width, new_height)),
                (target_width, target_height),
                color="black",
            )

            # Save the resized image
            resized_image_path = f"v1/Services/Story/Video_Creation/images/{i}.jpg"
            resized_img.save(resized_image_path)
            resized_image_paths.append(resized_image_path)
            i = i + 1

        return resized_image_paths

    except Exception as e:
        print("########### Error in resize_images function ##########")
        print(e)


def create_video(sentence_texts, audio_paths, image_paths):
    try:
        print("\nStart Compiling Video Clips........................................\n")
        # Create video clips from images and sentences
        video_clips = []
        for image_path, sentence_text, audio_path in zip(
            image_paths, sentence_texts, audio_paths
        ):
            # Load the audio clip for the current sentence
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration

            # Load the image clip
            image_clip = VideoFileClip(image_path)

            # Calculate the size and position for the text clip
            text_clip_width = image_clip.size[0]
            text_clip_height = 50
            text_clip_position = ("center", image_clip.size[1] - text_clip_height)

            # Create a text clip with the sentence and black background
            sentence_clip = TextClip(
                sentence_text,
                fontsize=15,
                color="yellow",
                size=(text_clip_width, text_clip_height),
                method="caption",
                bg_color="black",
            )
            sentence_clip = sentence_clip.set_duration(audio_duration)

            # Set the position of the text clip
            sentence_clip = sentence_clip.set_position(text_clip_position)

            # Overlay the text clip on the image clip
            video_clip = CompositeVideoClip(
                [image_clip.set_duration(audio_duration), sentence_clip]
            )
            video_clip = video_clip.set_audio(audio_clip)
            video_clips.append(video_clip)

        # Concatenate the video clips for final
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Write the final video to a file
        output_path = f"v1/Services/Story/Video_Creation/story_video.mp4"
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        copy_video_frontend = (
            f"D:/WorkPlace/Reserach Project/frontend/src/story_video.mp4"
        )
        shutil.copy(output_path, copy_video_frontend)

        return output_path

    except Exception as e:
        print("########### Error in create_video function ##########")
        print(e)
        return False


def extract_best_keyword_nltk(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)

    # Filter out stopwords (common words that are usually not meaningful)
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]

    # Part-of-speech tagging to identify nouns
    tagged_words = pos_tag(words)

    # Filter out only nouns (NN, NNS, NNP, NNPS)
    noun_words = [
        word for word, pos in tagged_words if pos in ["NN", "NNS", "NNP", "NNPS"]
    ]

    if not noun_words:
        # If there are no noun words, return the original input
        return sentence

    # Convert the list of noun words back into a sentence
    cleaned_sentence = " ".join(noun_words)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Calculate the TF-IDF scores for each word in the sentence
    tfidf_matrix = vectorizer.fit_transform([cleaned_sentence])

    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Find the word with the highest TF-IDF score
    max_tfidf_index = tfidf_matrix.toarray()[0].argmax()

    # Extract the best noun keyword
    best_noun_keyword = feature_names[max_tfidf_index]

    return best_noun_keyword


def change_image_style_to_cartoon(image_path):
    try:
        image = cv2.imread(image_path)

        # image_resized = cv2.resize(image, None, fx=0.5, fy=0.5)
        image_resized = image

        # Phase 2
        image_cleared = cv2.medianBlur(image_resized, 3)
        image_cleared = cv2.medianBlur(image_cleared, 3)
        image_cleared = cv2.medianBlur(image_cleared, 3)
        image_cleared = cv2.edgePreservingFilter(image_cleared, sigma_s=5)

        # Phase 3
        image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)

        for i in range(2):
            image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)

        for i in range(3):
            image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)

        # Phase 4
        gaussian_mask = cv2.GaussianBlur(image_filtered, (7, 7), 2)
        image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
        image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)

        # Save the cartoon image
        cv2.imwrite(image_path, image_sharp)

    except Exception as e:
        print("########### Error in cartoon image function ##########")
        print(e)
