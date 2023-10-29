from transformers import T5Tokenizer, T5ForConditionalGeneration

TOKENIZER_PATH = "v1/Services/ShortNote/TrainedModel"
MODEL_PATH = "v1/Services/ShortNote/TrainedModel"


def generate_summary_using_model(chunks, max_gen_length_model, min_gen_length_model):
    try:
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

        summaries = []
        for chunk in chunks:
            print("chunks....")
            print(chunk)

            inputs = tokenizer.encode(
                "summarize: " + chunk,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )
            summary_ids = model.generate(
                inputs,
                max_length=max_gen_length_model,
                min_length=min_gen_length_model,
                num_beams=4,
                temperature=1.0,
                early_stopping=True,
            )
            summarized_chunk = (
                tokenizer.decode(summary_ids[0])
                .replace("<pad>", "")
                .replace("</s>", "")
                .replace("–", "")
            )
            print("summarized....\n")
            print(summarized_chunk)
            summaries.append(summarized_chunk)

        final_summary = " ".join(summaries)
        print("final_summary")
        print(final_summary)

        return final_summary

    except Exception as e:
        print("Error occured in generate_summary_using_model function. Error is: ")
        print(e)
        return None
