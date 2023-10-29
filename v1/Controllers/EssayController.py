from v1.Services.Essay.AEG import predictscore
from v1.Services.Essay.AEG_v2 import predictscore_v2


def calculate_essay_score(topic: str, essay: str):
    print(topic, essay)
    res = predictscore(topic, essay)
    if res is not None and "score" in res:
        score_dataframe = res["score"]

        # Extract the score value from the DataFrame
        score = float(score_dataframe.iloc[0, 0]) if not score_dataframe.empty else 0.0

        feedback = res.get("feedback", "")
        return {"score": score, "feedback": feedback}
    else:
        return {"error": "Error in Predicting Score"}


def calculate_essay_score_v2(topic: str, essay: str):
    res = predictscore_v2(topic, essay)
    if res is not None:
        return res
    else:
        return {"error": "Error in Predicting Score"}
