# import modules
# import sys
# import re
# import keras
import pandas as pd
import string
import nltk
import spacy
import language_tool_python
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
)


from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from keras.models import load_model

nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stopwords = stopwords.words("english")
punctuations = string.punctuation

# loading model paths
model_topic1_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_1.h5"
model_topic2_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_2.h5"
model_topic3_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_3.h5"
model_topic4_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_4.h5"
model_topic5_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_5.h5"
model_topic6_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_6.h5"
model_topic7_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_7.h5"
model_topic8_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weights - 8-11\model_topic_8.h5"

# loading paths
training_features = "v1\Services\Essay\AEG_Files\training_features.pkl"
training_set_rel3 = r"v1\Services\Essay\AEG_Files\training_set_rel3.tsv"


valid_set_path = "v1\Services\Essay\AEG_Files\valid_set.tsv"
test_set_path = "v1\Services\Essay\AEG_Files\test_set.tsv"
# sys.path.append('AEG_Files')  # to get the bhkappa files
combo_set_path = "AEG_Files/combo_set.pkl"
wordvec_model = Word2Vec.load("v1\Services\Essay\AEG_Files\wordvec_model")
scaling_parameters_path = "v1\Services\Essay\AEG_Files\scaling_parameters.npy"


# function start here
def predictscore(topic: str, essay: str):
    try:
        # Define a essay to be predicted
        # Essay = ''' Dear Local Newspaper, @CAPS1 I have found that many experts say that computers do not benifit our society. In some cases this is true but in most cases studdies show that computers can help people. While nothing beats talking in person computers can get really close such examples are webcams or e-mail. @PERCENT1 of students who get good grades refer to reliable websites for reasearch or to help find good books. Also online catalouges or advertisments help the economy of stores worldwide. @CAPS2 people were not allowed to use computers most of the modern would not exist. @PERSON1 said that the best form of modern communication is the computer because of the ability to write, talk, or write back for much cheaper! Almost every single event i go to is planed on a computer by communication such as e-mail "@CAPS2 a student ever needs homework because lam out sick or needs help studying for a test then contact their teacher through the best form of communication for them always e-mail. Even the post office uses computers to get letters and boxes to people. The president of the post office, @PERSON2 said "@CAPS3 would be imposible to get mail to our coustmers @CAPS2 @CAPS3 were not for computers telling us where a zip code is or how heavy a box is." @CAPS4 that tell people what is happening around the world would not exist @CAPS2 @CAPS3 were not for the moder communication abilities that computer provid us. Because information can be commucated so quick. so can reasearch. When the country of @LOCATION2 took a pole @PERCENT2 of people used computer for any type of reasearch, of those @PERCENT3 were students currently in school and @PERCENT4 of them have good grades. When the same survey was taken in the @LOCATION1 @PERCENT5 of people used computers fore reasons and @PERCENT2 were students who had good grade @CAPS2 @CAPS3 were not posible for me to access documents in the @CAPS5 @CAPS6 online I probably would not have gotten an A+ on my @CAPS7 assignment! Could you amagine @CAPS2 suddenly your Newspaper reporters couldn't use the internet to work on their reports? The articles would probably be @NUM1 after the events occur. Most buissness, including the Newspaper, use the internet to advertise, shop, or read. The association of @ORGANIZATION1 reported that in @PERCENT1 of @ORGANIZATION1 used a website and of them @PERCENT5 were in good positions. The president of @CAPS8 @NUM2 imports said that they use an online catalouge because @CAPS3 is cheaper, but they can also promote that @CAPS3 is to save trees, or for the castomer's convinence. Small @ORGANIZATION1 can make websites to promote them selves and explain their star to potential coustomers. @PERSON3, the owner of @ORGANIZATION2's said that the internet saved her resturant. @CAPS2 @CAPS3 wer not for the internet @NUM3 more people would be jobless in @LOCATION3. In conclusion computer help everyday people and without them most convinences would not exist. They help communicate around the world. Computers help people reaserch subjects for school reports, and they make the current economy get better everyday. In moderation computers are the most useful tool out there.'''

        Essay = str(essay)

        if topic == "topic1":
            model_topic_path = model_topic1_path
        elif topic == "topic2":
            model_topic_path = model_topic2_path
        elif topic == "topic3":
            model_topic_path = model_topic3_path
        elif topic == "topic4":
            model_topic_path = model_topic4_path
        elif topic == "topic5":
            model_topic_path = model_topic5_path
        elif topic == "topic6":
            model_topic_path = model_topic6_path
        elif topic == "topic7":
            model_topic_path = model_topic7_path
        elif topic == "topic8":
            model_topic_path = model_topic8_path
        else:
            print("Topic is not there")
            return None

        training_set = pd.read_csv(
            training_set_rel3, sep="\t", encoding="ISO-8859-1"
        ).rename(
            columns={
                "essay_set": "topic",
                "domain1_score": "target_score",
                "domain2_score": "topic2_target",
            }
        )

        # use corrrect langauge to add matces, correction and corrected
        def correct_language(df):
            tool = language_tool_python.LanguageTool("en-US")

            df["matches"] = df["essay"].apply(lambda txt: tool.check(txt))
            df["corrections"] = df.apply(lambda l: len(l["matches"]), axis=1)
            df["corrected"] = df.apply(lambda l: tool.correct(l["essay"]), axis=1)

            return df

        # Define function to create averaged word vectors given a cleaned text.
        def create_average_vec(essay):
            average = np.zeros((text_dim,), dtype="float32")
            num_words = 0.0
            for word in essay.split():
                if word in wordvec_model.wv.key_to_index:
                    average = np.add(average, wordvec_model.wv[word])
                    num_words += 1.0
            if num_words != 0.0:
                average = np.divide(average, num_words)
            return average

        # converting to Data frame
        data = {"essay": [Essay], "topic": 1}
        df = pd.DataFrame(data)

        # add word count to df
        df["word_count"] = df["essay"].str.strip().str.split().str.len()

        # aplly correct_langauge to our df
        correct_language(df)

        # Generate additional features with spacy

        sents = []
        tokens = []
        lemma = []
        pos = []
        ner = []

        stop_words = set(STOP_WORDS)
        stop_words.update(punctuations)  # remove it if you need punctuation

        # suppress numpy warnings
        # np.warnings.filterwarnings("ignore")
        import spacy

        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")

        def process_essay(essay):
            if essay.is_parsed:
                return (
                    [e.text for e in essay],
                    [sent.text.strip() for sent in essay.sents],
                    [e.pos_ for e in essay],
                    [e.text for e in essay.ents],
                    [n.lemma_ for n in essay],
                )
            else:
                return None, None, None, None, None

        # Assuming df is your DataFrame containing the 'corrected' column
        tokens, sents, pos, ner, lemma = zip(
            *[process_essay(essay) for essay in nlp.pipe(df["corrected"])]
        )

        df["tokens"] = tokens
        df["lemma"] = lemma
        df["pos"] = pos
        df["sents"] = sents
        df["ner"] = ner

        ##checke the similarity
        """Choose arbitrary essay from highest available target_score for each topic.
        all other essays will be compared to these.
        The uncorrected essays will be used since the reference essays should have fewer errors.
        """
        reference_essays = {
            1: 161,
            2: 3022,
            3: 5263,
            4: 5341,
            5: 7209,
            6: 8896,
            7: 11796,
            8: 12340,
        }  # topic: essay_id

        references = {}

        nlp = spacy.load("en_core_web_sm")
        stop_words = set(STOP_WORDS)

        # generate nlp object for reference essays:
        for topic, index in reference_essays.items():
            references[topic] = nlp(training_set.iloc[index]["essay"])

        # generate document similarity for each essay compared to topic reference
        df["similarity"] = df.apply(
            lambda row: nlp(row["essay"]).similarity(references[row["topic"]]), axis=1
        )

        ## count various features
        df["token_count"] = df.apply(lambda x: len(x["tokens"]), axis=1)
        df["unique_token_count"] = df.apply(lambda x: len(set(x["tokens"])), axis=1)
        df["nostop_count"] = df.apply(
            lambda x: len([token for token in x["tokens"] if token not in stop_words]),
            axis=1,
        )
        df["sent_count"] = df.apply(lambda x: len(x["sents"]), axis=1)
        df["ner_count"] = df.apply(lambda x: len(x["ner"]), axis=1)
        df["comma"] = df.apply(lambda x: x["corrected"].count(","), axis=1)
        df["question"] = df.apply(lambda x: x["corrected"].count("?"), axis=1)
        df["exclamation"] = df.apply(lambda x: x["corrected"].count("!"), axis=1)
        df["quotation"] = df.apply(
            lambda x: x["corrected"].count('"') + x["corrected"].count("'"), axis=1
        )
        df["organization"] = df.apply(
            lambda x: x["corrected"].count(r"@ORGANIZATION"), axis=1
        )
        df["caps"] = df.apply(lambda x: x["corrected"].count(r"@CAPS"), axis=1)
        df["person"] = df.apply(lambda x: x["corrected"].count(r"@PERSON"), axis=1)
        df["location"] = df.apply(lambda x: x["corrected"].count(r"@LOCATION"), axis=1)
        df["money"] = df.apply(lambda x: x["corrected"].count(r"@MONEY"), axis=1)
        df["time"] = df.apply(lambda x: x["corrected"].count(r"@TIME"), axis=1)
        df["date"] = df.apply(lambda x: x["corrected"].count(r"@DATE"), axis=1)
        df["percent"] = df.apply(lambda x: x["corrected"].count(r"@PERCENT"), axis=1)
        df["noun"] = df.apply(lambda x: x["pos"].count("NOUN"), axis=1)
        df["adj"] = df.apply(lambda x: x["pos"].count("ADJ"), axis=1)
        df["pron"] = df.apply(lambda x: x["pos"].count("PRON"), axis=1)
        df["verb"] = df.apply(lambda x: x["pos"].count("VERB"), axis=1)
        df["noun"] = df.apply(lambda x: x["pos"].count("NOUN"), axis=1)
        df["cconj"] = df.apply(lambda x: x["pos"].count("CCONJ"), axis=1)
        df["adv"] = df.apply(lambda x: x["pos"].count("ADV"), axis=1)
        df["det"] = df.apply(lambda x: x["pos"].count("DET"), axis=1)
        df["propn"] = df.apply(lambda x: x["pos"].count("PROPN"), axis=1)
        df["num"] = df.apply(lambda x: x["pos"].count("NUM"), axis=1)
        df["part"] = df.apply(lambda x: x["pos"].count("PART"), axis=1)
        df["intj"] = df.apply(lambda x: x["pos"].count("INTJ"), axis=1)

        pd.set_option("display.max_columns", None)  # To display all columns

        """### Generate word embeddings with Word2Vec"""

        # craete clened vectors
        def cleanup_essays(essays, logging=False):
            texts = []
            counter = 1
            for essay in essays.corrected:
                if counter % 2000 == 0 and logging:
                    print("Processed %d out of %d documents." % (counter, len(essays)))
                counter += 1
                essay = nlp(essay, disable=["parser", "ner"])
                tokens = [
                    tok.lemma_.lower().strip()
                    for tok in essay
                    if tok.lemma_ != "-PRON-"
                ]
                tokens = [
                    tok
                    for tok in tokens
                    if tok not in stopwords and tok not in punctuations
                ]
                tokens = " ".join(tokens)
                texts.append(tokens)
            return pd.Series(texts)

        # pass the essay into cleanup_essays
        train_cleaned = cleanup_essays(df, logging=True)

        from gensim.models.word2vec import Word2Vec

        wordvec_model = Word2Vec.load("v1\Services\Essay\AEG_Files\wordvec_model")

        # Create word vectors
        text_dim = 300
        cleaned_vec = np.zeros((df.shape[0], text_dim), dtype="float32")
        for i in range(len(train_cleaned)):
            cleaned_vec[i] = create_average_vec(train_cleaned[i])

        print(
            "Word vectors for all essays in the training data set are of shape:",
            cleaned_vec.shape,
        )

        # Add feature list
        # Read generated features from file:

        # Use select features from Gini feature importances
        feature_list = [
            "word_count",
            "corrections",
            "similarity",
            "token_count",
            "unique_token_count",
            "nostop_count",
            "sent_count",
            "ner_count",
            "comma",
            "question",
            "exclamation",
            "quotation",
            "organization",
            "caps",
            "person",
            "location",
            "money",
            "time",
            "date",
            "percent",
            "noun",
            "adj",
            "pron",
            "verb",
            "cconj",
            "adv",
            "det",
            "propn",
            "num",
            "part",
            "intj",
        ]

        additional_features = df[feature_list]

        """###Loading scalling parameters to scale"""

        # loading the scallng parameter from training data to scale
        loaded_data = np.load(scaling_parameters_path)
        mean_values, std_values = loaded_data
        additional_features = (additional_features - mean_values) / std_values
        additional_features = np.array([additional_features])
        additional_features = additional_features.reshape(1, 31)

        # prepare input to model to pass into model
        input = pd.concat(
            [pd.DataFrame(additional_features), pd.DataFrame(cleaned_vec)], axis=1
        )

        """###Load model to get prediction

        """

        # load model & prediction
        model = load_model(model_topic_path)
        y_pred = pd.DataFrame(model.predict(input).reshape(-1))
        y_pred = np.round(y_pred)

        """### Feedback Generation"""

        def provide_feedback(row):
            feedback = f"Your essay submitted for topic {row['topic'].values[0]} has the following details:\n"
            feedback += f"- Word Count: {row['word_count'].values[0]} words\n"
            feedback += (
                f"- Grammar Corrections: {row['corrections'].values[0]} corrections\n"
            )
            feedback += f"- Total Sentences: {row['sent_count'].values[0]} sentences\n"
            feedback += f"- Unique Token Count: {row['unique_token_count'].values[0]} unique tokens\n"
            feedback += f"- Adjective Count: {row['adj'].values[0]} adjectives\n"
            feedback += f"- noun Count: {row['noun'].values[0]} noun\n"
            feedback += f"- Verb Count: {row['verb'].values[0]} verbs\n"

            return feedback

        print(provide_feedback(df))
        feedback = provide_feedback(df)

        result = {"score": y_pred, "feedback": feedback}
        return result

    except Exception as e:
        print("Error occured in predictscore function. Error is: ")
        print(e)
        return None
