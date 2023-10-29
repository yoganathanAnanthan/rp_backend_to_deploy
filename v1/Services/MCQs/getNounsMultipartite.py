import nltk
nltk.download('stopwords')
# from nltk.corpus import stopwords
# import string
# import pke
# import traceback
import spacy
from fuzzywuzzy import fuzz


def get_nouns_multipartite(content):
    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")

    # Process the context using spaCy
    doc = nlp(content)

    # Initialize a list to store the extracted keywords
    keywords = []

    # Extract entities as keywords
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'GPE', 'ORG']:  # Person names, place names, and organization names
            keywords.append(ent.text)

    # Extract noun chunks as keywords
    for chunk in doc.noun_chunks:
        if chunk.root.text.lower() not in nlp.Defaults.stop_words:
            keywords.append(chunk.text)

    # Perform word frequency analysis
    word_frequencies = {}
    for token in doc:
        if token.is_alpha and not token.is_stop:
            if token.text.lower() in word_frequencies:
                word_frequencies[token.text.lower()] += 1
            else:
                word_frequencies[token.text.lower()] = 1

    # Add high-frequency words as keywords
    threshold = 2  # Minimum frequency to consider a word as a keyword
    for word, frequency in word_frequencies.items():
        if frequency >= threshold:
            keywords.append(word)

    # Filter out stopwords from the keywords
    keywords = [keyword for keyword in keywords if keyword.lower() not in nlp.Defaults.stop_words]

    # Remove similar keywords
    unique_keywords = []
    for keyword in keywords:
        is_similar = False
        for unique_keyword in unique_keywords:
            similarity_ratio = fuzz.ratio(keyword.lower(), unique_keyword.lower())
            if similarity_ratio >= 80:  # Adjust the threshold as needed
                is_similar = True
                break
        if not is_similar:
            unique_keywords.append(keyword)

    return unique_keywords