import openai

# Set up your OpenAI API key
openai.api_key = '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'
def generate_distractors(question, correct_answer):
    prompt = f"Question: {question}\nAnswer: {correct_answer}\nDistractors:"

    # Generate distractors using open AI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=4,
        stop=None,
    )

    # Extract the generated distractors from the response
    distractors = [choice['text'].strip() for choice in response.choices]
    print(distractors)

    # Remove any numbering and leading dashes from the distractors
    cleaned_distractors = []
    for distractor in distractors:
        lines = distractor.split("\n")
        cleaned_lines = [
            line.split(". ")[-1] if line.strip().lstrip("1234567890").strip().startswith(".") else line.lstrip("-").strip() if line.strip().startswith("-") else line.lstrip("1234567890. ").strip()
            for line in lines if line.strip()
        ]

        cleaned_distractors.extend([d.strip() for d in "; ".join(cleaned_lines).split(";") if d.strip()])

    return cleaned_distractors