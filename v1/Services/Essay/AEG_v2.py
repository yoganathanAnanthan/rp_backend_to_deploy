# import modules
# import sys
# import keras
import pandas as pd
import string
import re
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
from collections import Counter
import re

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
model_topic4_path = "v1\Services\Essay\AEG_Files\AEG_Turanga 8-11 Files\AEG Model_Weightsopic_details - 8-11\model_topic_4.h5"
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
combo_set_path = "v1\Services\Essay\AEG_Files\combo_set.pkl"
wordvec_model = Word2Vec.load("v1\Services\Essay\AEG_Files\wordvec_model")
scaling_parameters_path = "v1\Services\Essay\AEG_Files\scaling_parameters.npy"


# function start here
def predictscore_v2(topic: str, essay: str):
    try:
        # Define a essay to be predicted
        # Essay = ''' Dear Local Newspaper, @CAPS1 I have found that many experts say that computers do not benifit our society. In some cases this is true but in most cases studdies show that computers can help people. While nothing beats talking in person computers can get really close such examples are webcams or e-mail. @PERCENT1 of students who get good grades refer to reliable websites for reasearch or to help find good books. Also online catalouges or advertisments help the economy of stores worldwide. @CAPS2 people were not allowed to use computers most of the modern would not exist. @PERSON1 said that the best form of modern communication is the computer because of the ability to write, talk, or write back for much cheaper! Almost every single event i go to is planed on a computer by communication such as e-mail "@CAPS2 a student ever needs homework because lam out sick or needs help studying for a test then contact their teacher through the best form of communication for them always e-mail. Even the post office uses computers to get letters and boxes to people. The president of the post office, @PERSON2 said "@CAPS3 would be imposible to get mail to our coustmers @CAPS2 @CAPS3 were not for computers telling us where a zip code is or how heavy a box is." @CAPS4 that tell people what is happening around the world would not exist @CAPS2 @CAPS3 were not for the moder communication abilities that computer provid us. Because information can be commucated so quick. so can reasearch. When the country of @LOCATION2 took a pole @PERCENT2 of people used computer for any type of reasearch, of those @PERCENT3 were students currently in school and @PERCENT4 of them have good grades. When the same survey was taken in the @LOCATION1 @PERCENT5 of people used computers fore reasons and @PERCENT2 were students who had good grade @CAPS2 @CAPS3 were not posible for me to access documents in the @CAPS5 @CAPS6 online I probably would not have gotten an A+ on my @CAPS7 assignment! Could you amagine @CAPS2 suddenly your Newspaper reporters couldn't use the internet to work on their reports? The articles would probably be @NUM1 after the events occur. Most buissness, including the Newspaper, use the internet to advertise, shop, or read. The association of @ORGANIZATION1 reported that in @PERCENT1 of @ORGANIZATION1 used a website and of them @PERCENT5 were in good positions. The president of @CAPS8 @NUM2 imports said that they use an online catalouge because @CAPS3 is cheaper, but they can also promote that @CAPS3 is to save trees, or for the castomer's convinence. Small @ORGANIZATION1 can make websites to promote them selves and explain their star to potential coustomers. @PERSON3, the owner of @ORGANIZATION2's said that the internet saved her resturant. @CAPS2 @CAPS3 wer not for the internet @NUM3 more people would be jobless in @LOCATION3. In conclusion computer help everyday people and without them most convinences would not exist. They help communicate around the world. Computers help people reaserch subjects for school reports, and they make the current economy get better everyday. In moderation computers are the most useful tool out there.'''

        Essay = str(essay)

        if topic == "topic1":
            model_topic_path = model_topic1_path 
            topic_no =1
        elif topic == "topic2":
            model_topic_path = model_topic2_path
            topic_no =2
        elif topic == "topic3":
            model_topic_path = model_topic3_path
            topic_no =3
        elif topic == "topic4":
            model_topic_path = model_topic4_path
            topic_no =4
        elif topic == "topic5":
            model_topic_path = model_topic5_path
            topic_no =5
        elif topic == "topic6":
            model_topic_path = model_topic6_path
            topic_no =6
        elif topic == "topic7":
            model_topic_path = model_topic7_path
            topic_no =7
        elif topic == "topic8":
            model_topic_path = model_topic8_path
            topic_no =8
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
        data = {"essay": [Essay], "topic": topic_no}
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
        y_pred1 = y_pred
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

        # Additional codes started here 10/27/2023

        ###modified updated feedback statistics 10/27/2023
        def provide_feedback_updated(row):
            feedback = f"Your essay submitted for topic {row['topic'].values[0]} has the following additional details:\n"
            feedback += f"- Word Count of the essay: {row['word_count'].values[0]} \n"
            feedback += f"- Total Sentences used: {row['sent_count'].values[0]} \n"
            feedback += f"- Similarity of your essay compared to the highest graded essay in the corpus: {round(row['similarity'].values[0] * 100, 2)}%\n"
            feedback += f"- Adjective Count: {row['adj'].values[0]} \n"
            feedback += f"- Questions in the essay: {row['question'].values[0]} \n"
            feedback += f"- Quotations in the essay: {round(row['quotation'].values[0] / 2, 0)} \n"
            feedback += (
                f"- Exclamation used in the essay: {row['exclamation'].values[0]} \n"
            )
            feedback += f"- Comma's used in the essay: {row['comma'].values[0]} \n"
            feedback += f"- Named Entities Count: {row['ner_count'].values[0]} \n"
            feedback += f"- Locations used in the essay: {row['location'].values[0]} \n"
            feedback += (
                f"- Person names used in the essay: {row['person'].values[0]} \n"
            )
            feedback += f"- Organization names used in the essay: {row['organization'].values[0]} \n"
            return feedback

        ###End of modified updated feedback statistics 10/27/2023

        ### most common word calculation 10/27/2023
        words = re.findall(r"\b\w+\b", Essay.lower())
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        most_common_bigram, most_common_count = bigram_counts.most_common(1)[0]
        ### end of most common word calculation 10/27/2023

        # word diversity calculation 10/27/2023
        word_diversity = df["unique_token_count"].values / df["word_count"].values * 100
        word_diversity = word_diversity[0]
        word_diversity = np.round(word_diversity, 2)
        word_diversity_per = str(word_diversity) + "%"
        word_diversity_per
        print("sasas")
        print(word_diversity_per)
        print(word_diversity)

        ###end of word diversity calculation 10/27/2023

        ###modified feedback geneartion primary 10/27/2023
        # def provide_feedback_General(row):
        #     feedback = f"Thank you for submitting your paper for topic {row['topic'].values[0]} Below you will find a few suggestions for how to strengthen your writing during the revision process.\n"
        #     if word_diversity < 70:
        #         feedback += f"Your essay has {row['word_count'].values[0]} total words and {row['unique_token_count'].values[0]} unique words, resulting in a diversity of {word_diversity}%.\nI recommend focusing on expanding your vocabulary. For instance,\nyour two most common words are {most_common_bigram}. Try using alternatives from a thesaurus. Here's a source to learn: https://www.thesaurus.com/browse/have.\n"
        #         if int(row["corrections"].values[0]) > 10:
        #             feedback += f"Your essay contains a total of {row['corrections'].values[0]} grammar errors. To improve your score, it's essential to minimize these grammatical mistakes."
        #         else:
        #             feedback += f"Your essay demonstrates a strong command of grammar with minimal errors of {row['corrections'].values[0]} , which is commendable"
        #     else:
        #         feedback += f"Your essay has {row['word_count'].values[0]} total words and {row['unique_token_count'].values[0]} unique words, resulting in a Diversity of {word_diversity}%. Your vocabulary is in good shape! Keep up the good work!\n"
        #         if int(row["corrections"].values[0]) > 10:
        #             feedback += f"Your essay contains a total of {row['corrections'].values[0]} grammar errors. To improve your score, it's essential to minimize these grammatical mistakes."
        #         else:
        #             feedback += f"Your essay demonstrates a strong command of grammar with minimal errors of {row['corrections'].values[0]} , which is commendable"
        #     return feedback

        ###modified feedback geneartion primary 1 10/27/2023
        def provide_feedback_General_1(row):
            feedback = f"Thank you for submitting your paper for topic {row['topic'].values[0]} Below you will find a few suggestions for how to strengthen your writing during the revision process."
            return feedback 
        
        ###modified feedback geneartion primary 2 10/27/2023
        def provide_feedback_General_2(row):
            if word_diversity < 75:
                feedback = f"Your essay has {row['word_count'].values[0]} total words and {row['unique_token_count'].values[0]} unique words, resulting in a diversity of {word_diversity}%. I recommend focusing on expanding your vocabulary. For instance,\nyour two most common words are {most_common_bigram}. Try using alternatives from a thesaurus. Here's a source to learn: https://www.thesaurus.com/browse/have.\n"
            else:
                feedback = f"Your essay has {row['word_count'].values[0]} total words and {row['unique_token_count'].values[0]} unique words, resulting in a Diversity of {word_diversity}%. Your vocabulary is in good shape! Keep up the good work!\n"                
            return feedback 
        
        ###modified feedback geneartion primary 3 10/27/2023
        def provide_feedback_General_3(row):
            if int(row['corrections'].values[0]) > 10:
                feedback = f"Your essay contains a total of {row['corrections'].values[0]} grammar errors. To improve your score, it's essential to minimize these grammatical mistakes."
            else:
                feedback = f"Your essay demonstrates a strong command of grammar with minimal errors of {row['corrections'].values[0]} , which is commendable"
            
            return feedback

        ###end of modified feedback geneartion primary 10/27/2023

        ### Scaled score impelentation 10/27/2023
        topic_details = [
            {"topic": 1, "count": 12, "min": 2, "max": 12},
            {"topic": 2, "count": 6, "min": 1, "max": 6},
            {"topic": 3, "count": 3, "min": 0, "max": 3},
            {"topic": 4, "count": 3, "min": 0, "max": 3},
            {"topic": 5, "count": 4, "min": 0, "max": 4},
            {"topic": 6, "count": 4, "min": 0, "max": 4},
            {"topic": 7, "count": 24, "min": 0, "max": 30},
            {"topic": 8, "count": 60, "min": 0, "max": 60},
        ]

        def scale_score(topic, score):
            topic_info = next((t for t in topic_details if t["topic"] == topic), None)
            if topic_info is not None:
                min_score, max_score = topic_info["min"], topic_info["max"]
                scaled_score = (score - min_score) / (max_score - min_score)
                return scaled_score
            else:
                return None

        topic = df["topic"].values[0]
        input_score = np.round(y_pred1.iloc[0, 0], 2)
        scaled_score = scale_score(topic, input_score)
        scaled_score = scaled_score * 100
        scaled_score = f"{scaled_score:.0f}%"
        # print(f"Scaled score {scaled_score:.0f}%")
        ####End of Scaled score impelentation 10/27/2023

        ###Return grammer corrected essay 10/27/2023
        def grammer_corrected():
            return df["corrected"].values[0]

        ### end of Return grammer corrected essay 10/27/2023

        ### view refeence essay and score 10/27/2023
        def view_reference_essay(topic, topic_details):
            if topic in references:
                for item in topic_details:
                    if item["topic"] == topic:
                        count = item["count"]
                        break
                print(
                    f"Highest scored essay for Topic {topic} in the corpus (Score: {count}):"
                )
                return references[topic].text
            else:
                print(f"No reference essay found for Topic {topic}.")
                return None

        essay_topic = df["topic"].values[0]
        # view_reference_essay(essay_topic, topic_details)
        ### end of  view refeence essay and score 10/27/2023

        # End of the all additional code 10/27/2023

        format_score = float(y_pred.iloc[0, 0]) if not y_pred.empty else 0.0
        feedback = provide_feedback_updated(df)
        referenc_essay = view_reference_essay(essay_topic, topic_details)
        grammer = grammer_corrected()
        print(topic)
        print(y_pred)

        result = {
            "score": format_score,
            "feedback": feedback,
            "scaled_score": scaled_score,
            "referenc_essay": referenc_essay,
            "grammer": grammer,
            "general_feedback_1": provide_feedback_General_1(df),
            "general_feedback_2": provide_feedback_General_2(df),
            "general_feedback_3": provide_feedback_General_3(df),

         
        }
        return result

    except Exception as e:
        print("Error occured in predictscore function. Error is: ")
        print(e)
        return None
    
# predictscore_v2('topic2','''Dear local Newspaper @CAPS1 a take all your computer and given to the people around the world for the can stay in their houses chating with their family and friend. Computers help people around the world to connect with other people computer help kids do their homework and look up staff that happen around the world.''')


# output
# predictscore('topic1','''Dear @ORGANIZATION1, The computer blinked to life and an image of a blonde haired girl filled the screen. It was easy to find out how life was in @LOCATION2, thanks to the actual @CAPS1 girl explaining it. Going to the library wouldn't have filled one with this priceless information and human interection. Computers are a nessessity of life if soceity wishes to grow and expand. They should be supported because they teach hand eye coordination, give people the ability to learn about faraway places, and allow people to talk to others online. Firstly, computers help teach hand eye coordination. Hand-eye coordination is a useful ability that is usod to excel in sports. In a recent survey, @PERCENT1 of kids felt their hand eye coordination improves after computer use. Even a simple thing like tying can build up this skill. Famous neurologist @CAPS2 @PERSON1 stated in an article last week that, "@CAPS3 and computer strength the @CAPS2. When on the computer, you automatically process what the eyes see into a command for your hands." @CAPS4 hand eye coordination can improve people in sports such as baseball and basketball. If someone wan't to become better in these sports, all they'd need to do was turn on the computer. Once people become better at sports, they're more likely to play them and become more healthy. In reality, computers can help with exercising instead of decreasing it. Additionaly, computers allow people to access information about faraway places and people. If someone wanted to reasearch @LOCATION1, all they'd need to do was type in a search would be presented to them in it would link forever to search through countless things. Also, having the ability to learn about cultures can make peole peole and their cultures, they understand others something. Increase tolerance people are. Computers are a resourceful tool that they can help people in every different aspect of life. Lastly, computer and in technology can allow people to chat. Computer chat and video chat can help the all different nations. Bring on good terms places other than can help us understand story comes out about something that happend in @LOCATION3, people can just go on their computer and ask an actual @LOCATION3 citizen their take on the matter. Also, video chat and online conversation can cut down on expensive phone bills. No one wants to pay more than they have to in this economy. Another good point is that you can acess family members you scaresly visit. It can help you connect within your own family more. Oviously, computers are a useful aid in todays era. their advancements push the world foreward to a better place. Computers can help people because they help teach handeye coordination, give people the bility to learn about faraway places and people, and allow people to talk online with others. Think of a world with no computers or technologicall advancements. The world would be sectored and unified, contact between people scare, and information even. The internet is like thousands or librarys put together. Nobody would know much about other nations and news would travel slower. Is that the kind of palce you want people to live in?''')
# predictscore("topic1", "my named is donky")
