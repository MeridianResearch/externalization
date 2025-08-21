import os
import time
import pandas as pd
from typing import Optional
from openai import OpenAI

client = OpenAI()

DATASET = "results_and_data/early_exit_sft_dataset/test/validation.csv"
OUTPUT  = "results_and_data/early_exit_sft_dataset/test/validation_w_answers.csv"

SYSTEM_PROMPT = "You are a precise reasoning assistant. Given a short story and a question, infer the correct answer. Return ONLY the answer as a short phrase (no punctuation beyond what's necessary, no preamble)."

USER_TEMPLATE = """Story:
{story}

Question:
{question}
"""

def ask_model(story: str, question: str) -> str:
    resp = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        instructions=SYSTEM_PROMPT,
        input=USER_TEMPLATE.format(story=story, question=question)
    )
    out = resp.output_text
    if out:
        return out
    return "[ERROR: empty response]"


def main():

    df = pd.read_csv(DATASET, dtype=str).fillna("") 
    answers = []
    for _, row in df.iterrows():
        story = row["story"]
        question = row["question"]
        try:
            ans = ask_model(story, question)
        except Exception as e:
            ans = f"[ERROR: {e}]"
        answers.append(ans)
        print(ans)

    df["answer"] = answers

    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} rows with answers to: {OUTPUT}")

if __name__ == "__main__":
    main()
