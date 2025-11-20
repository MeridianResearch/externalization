"""
Minimal evaluator for computational savings experiments.

Reads the specified CSV file, evaluates up to N rows of both
`generation_text` and `normal_generation_text` using GPT-5 as an automated
judge, and writes the results into a copy of the CSV named
`run_log-7_Evals.csv`. Only the required logic is implemented.
"""

import argparse
import asyncio
import os
from typing import Dict, Tuple

import certifi
import pandas as pd

from inspect_ai.model import get_model as get_inspect_model

ASSISTANT_TAGS = [
    "<｜Assistant｜>",
    "<|Assistant|>",
    "<|assistant|>",
    "<assistant>",
    "< Assistant >",
    "<Assistant>",
]
DEFAULT_INPUT_PATH = "early_exit_teacher/visualizations/run_log-7.csv"
DEFAULT_OUTPUT_PATH = "early_exit_teacher/visualizations/run_log-7_Evals.csv"
DEFAULT_ROWS = 1

EVAL_PROMPT_TEMPLATE = """
You are evaluating the coherence and quality of an AI assistant's chain of thought and response.
Allow exploratory/meandering reasoning as long as it stays internally consistent.

Original Question: {prompt}

AI Response: {response_text}

Please evaluate this response on the following criteria:
1. Coherence and logical flow (1-10 scale) Does the reasoning follow a sensible progression, even if unfinished?
2. Completeness of the reasoning (1-10 scale) Does it reach a correct and explicit chain of thought? If partial but on-track and relevant to answer, award mid-range (4–7).
3. Clarity and readability (1-10 scale) Is it easy to follow? Minor meandering is okay if readable.
4. Absence of repetition or errors (1-10 scale) Penalize contradictions, factual mistakes about the prompt, or heavy repetition.

Rate each criterion and provide an overall score from 1-10:
- 1: major breakdown (nonsensical, off-topic)
- 4: noticeable issues but some useful reasoning
- 7: generally solid, with minor flaws or cut-offs
- 10: excellent, complete, and polished

Meta / Wrapper Policy:
- The evaluation input may include wrapper/markup such as: angle-bracket role tags (e.g., <｜User｜>, <｜Assistant｜>) and <think>.
- These wrappers are expected and should not reduce scores for Clarity, Coherence, or No Repetition.

Format your response as:
Coherence: X/10
Completeness: X/10
Clarity: X/10
No Repetition: X/10
Overall: X/40

Brief explanation: [your reasoning]
""".strip()


def ensure_ssl_certificate() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["SSL_CERT_FILE"] = certifi.where()


def text_after_assistant_tag(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    lowered = text_str.lower()
    tag_index = -1
    tag_length = 0

    for tag in ASSISTANT_TAGS:
        idx = lowered.rfind(tag.lower())
        if idx != -1:
            tag_index = idx
            tag_length = len(tag)
            break

    if tag_index == -1:
        return text_str.strip()

    start_index = tag_index + tag_length
    assistant_text = text_str[start_index:].lstrip()

    # Stop at next role tag if present
    stop_tags = ["<｜User｜>", "<|User|>", "<|user|>", "<user>", "< User >", "<User>"]
    stop_index = -1
    for stag in stop_tags:
        idx = assistant_text.find(stag)
        if idx != -1:
            if stop_index == -1 or idx < stop_index:
                stop_index = idx
    
    if stop_index != -1:
        assistant_text = assistant_text[:stop_index].strip()

    return assistant_text


def build_eval_prompt(prompt_text: str, response_text: str) -> str:
    return EVAL_PROMPT_TEMPLATE.format(prompt=prompt_text, response_text=response_text)


def print_payload(label: str, prompt_text: str, response_text: str) -> None:
    print(f"\n=== PAYLOAD ({label}) ===")
    print("Prompt supplied to judge:")
    print(prompt_text if prompt_text else "[EMPTY PROMPT]")
    print("Response after <｜Assistant｜> tag:")
    print(response_text if response_text else "[EMPTY RESPONSE]")


def parse_scores(eval_text: str) -> Dict[str, float]:
    scores = {
        "coherence": None,
        "completeness": None,
        "clarity": None,
        "no_repetition": None,
        "average": None,
        "raw_response": eval_text,
    }
    try:
        for line in eval_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Coherence:"):
                scores["coherence"] = int(stripped.split(":")[1].split("/")[0].strip())
            elif stripped.startswith("Completeness:"):
                scores["completeness"] = int(stripped.split(":")[1].split("/")[0].strip())
            elif stripped.startswith("Clarity:"):
                scores["clarity"] = int(stripped.split(":")[1].split("/")[0].strip())
            elif stripped.startswith("No Repetition:"):
                scores["no_repetition"] = int(stripped.split(":")[1].split("/")[0].strip())
            elif stripped.startswith("Overall:"):
                overall = int(stripped.split(":")[1].split("/")[0].strip())
                scores["average"] = overall / 4.0
        if scores["average"] is None:
            numeric_parts = [
                scores["coherence"],
                scores["completeness"],
                scores["clarity"],
                scores["no_repetition"],
            ]
            numeric_parts = [part for part in numeric_parts if part is not None]
            if numeric_parts:
                scores["average"] = sum(numeric_parts) / len(numeric_parts)
    except Exception as error:
        scores["raw_response"] = f"Failed to parse: {error}\n{eval_text}"
    return scores


async def query_judge(
    model, prompt_text: str, label: str
) -> Tuple[Dict[str, float], str]:
    print(f"\n--- GPT-5 REQUEST ({label}) ---")
    print(prompt_text)
    eval_result = await model.generate(prompt_text)
    eval_text = eval_result.completion
    print(f"--- GPT-5 RESPONSE ({label}) ---")
    print(eval_text)
    parsed_scores = parse_scores(eval_text)
    return parsed_scores, eval_text


async def process_rows(
    dataframe: pd.DataFrame, max_rows: int, model
) -> pd.DataFrame:
    processed_df = dataframe.copy()
    target_rows = min(max_rows, len(processed_df))

    new_columns = [
        "ee_coherence",
        "ee_completeness",
        "ee_clarity",
        "ee_no_repetition",
        "ee_average",
        "normal_coherence",
        "normal_completeness",
        "normal_clarity",
        "normal_no_repetition",
        "normal_average",
    ]
    for column in new_columns:
        if column not in processed_df.columns:
            processed_df[column] = pd.NA

    for idx in range(target_rows):
        row = processed_df.iloc[idx]
        prompt_text = str(row.get("prompt", ""))

        ee_text = text_after_assistant_tag(row.get("generation_text", ""))
        if ee_text:
            ee_prompt = build_eval_prompt(prompt_text, ee_text)
            print_payload("generation_text", prompt_text, ee_text)
            ee_scores, _ = await query_judge(model, ee_prompt, "generation_text")
            processed_df.at[
                processed_df.index[idx], "ee_coherence"
            ] = ee_scores["coherence"]
            processed_df.at[
                processed_df.index[idx], "ee_completeness"
            ] = ee_scores["completeness"]
            processed_df.at[
                processed_df.index[idx], "ee_clarity"
            ] = ee_scores["clarity"]
            processed_df.at[
                processed_df.index[idx], "ee_no_repetition"
            ] = ee_scores["no_repetition"]
            processed_df.at[
                processed_df.index[idx], "ee_average"
            ] = ee_scores["average"]

        normal_text = text_after_assistant_tag(row.get("normal_generation_text", ""))
        if normal_text:
            normal_prompt = build_eval_prompt(prompt_text, normal_text)
            print_payload("normal_generation_text", prompt_text, normal_text)
            normal_scores, _ = await query_judge(model, normal_prompt, "normal_generation_text")
            processed_df.at[
                processed_df.index[idx], "normal_coherence"
            ] = normal_scores["coherence"]
            processed_df.at[
                processed_df.index[idx], "normal_completeness"
            ] = normal_scores["completeness"]
            processed_df.at[
                processed_df.index[idx], "normal_clarity"
            ] = normal_scores["clarity"]
            processed_df.at[
                processed_df.index[idx], "normal_no_repetition"
            ] = normal_scores["no_repetition"]
            processed_df.at[
                processed_df.index[idx], "normal_average"
            ] = normal_scores["average"]

    return processed_df


async def main():
    parser = argparse.ArgumentParser(description="Evaluate run log rows with GPT-5 judge.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, help="Input CSV path.")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path (copy with evaluations).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of rows to evaluate (default: 1).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    ensure_ssl_certificate()

    dataframe = pd.read_csv(args.input)
    model = get_inspect_model("openai/gpt-5")
    processed = await process_rows(dataframe, args.rows, model)
    processed.to_csv(args.output, index=False)
    print(f"\nSaved evaluations to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

