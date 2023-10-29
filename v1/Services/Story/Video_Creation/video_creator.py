import v1.Services.Story.Video_Creation.services as services


def make_story_as_video(english_text):
    print("\nStart Making Video Progrss........................................\n")
    english_sentences_list = services.split_sentences(english_text)

    # Generate audio for the each sentences
    audio_paths = []
    i = 1
    for sentence in english_sentences_list:
        audio_path = services.text_to_audio(
            sentence, "en", f"v1/Services/Story/Video_Creation/audios/{i}.mp3"
        )

        audio_paths.append(audio_path)
        i = i + 1

    # Download image and resized it
    j = 1
    image_paths = []
    for sentence in english_sentences_list:
        image_path = services.download_unsplash_image(
            sentence, f"v1/Services/Story/Video_Creation/images/{j}.jpg"
        )
        image_paths.append(image_path)
        j = j + 1

    resized_image_paths = services.resize_images(image_paths)

    return services.create_video(
        english_sentences_list, audio_paths, resized_image_paths
    )


def make_story_as_video_v2(english_text):
    print("\nStart Making Video Progrss........................................\n")
    english_sentences_list = services.split_sentences(english_text)

    # Generate audio for the each sentences
    audio_paths = []
    i = 1
    for sentence in english_sentences_list:
        audio_path = services.text_to_audio(
            sentence, "en", f"v1/Services/Story/Video_Creation/audios/{i}.mp3"
        )

        audio_paths.append(audio_path)
        i = i + 1

    # Download image and resized it
    j = 1
    image_paths = []
    for sentence in english_sentences_list:
        keyword = services.extract_best_keyword_nltk(sentence)
        print(f"nltk:{keyword}")
        image_path = services.download_unsplash_image(
            keyword, f"v1/Services/Story/Video_Creation/images/{j}.jpg"
        )
        services.change_image_style_to_cartoon(image_path)
        image_paths.append(image_path)
        j = j + 1

    resized_image_paths = services.resize_images(image_paths)

    return services.create_video(
        english_sentences_list, audio_paths, resized_image_paths
    )
