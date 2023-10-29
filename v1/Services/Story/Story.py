# Generation Code II
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Better Default Value
max_length = 256
max_story_length = 512
temperature = 0.7
repetition_penalty = 1.2

# max_length = 512
# max_story_length = 1024
# temperature = 0.7
# repetition_penalty = 1.2


def generate_story(
    prompt,
    model_name,
):
    try:
        print("\nStarted generating story............\n")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_story = tokenizer.decode(
                model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )[0],
                skip_special_tokens=True,
            )

        # Ensure the story does not exceed the specified maximum length
        if len(generated_story) > max_story_length:
            generated_story = generated_story[:max_story_length]

        print("\nGenerated Story:\n")
        print(generated_story)
        return generated_story

    except Exception as e:
        print("Error in generate_story function")
        print(e)
        return False
