import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "v1/Services/MCQs/TrainedModel/model"
tokenizer_path = "v1/Services/MCQs/TrainedModel/tokenizer"

question_model = T5ForConditionalGeneration.from_pretrained(model_path)
question_tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
question_model = question_model.to(device)


def get_question(context, answer):
    model = question_model
    tokenizer = question_tokenizer
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text,
        max_length=384,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=72,
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question
