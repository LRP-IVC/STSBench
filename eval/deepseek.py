import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .consts import PredictionKeys
from .eval import evaluate


def infer(args):
    api_key = os.environ.get(args.env_api_key, None)
    assert api_key is not None, (
        "Missing [OPENAI | DEEPSEEK]_API_KEY environment variable add it with `export [OPENAI | DEEPSEEK]_API_KEY=...`"
    )

    client = OpenAI(api_key=api_key, base_url=args.openai_base_url)

    with open(args.input_path, "r") as f:
        data = json.load(f)

    all_answers = []
    for qa in tqdm(data):
        prompt = qa["prompt"]
        sys_prompt, usr_prompt = prompt.split("|")  # a hacky way how we do it

        while True:  # make sure that each prompt gets an answer
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    stream=False,
                )
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                print("Waiting for 20 seconds to retry...")
                time.sleep(20)
                continue

            time.sleep(args.timeout)
            break

        all_answers.append(
            {
                PredictionKeys.MODEL_PREDICTION: response.choices[0].message.content,
                PredictionKeys.GT_LETTER: qa["answer_letter"],
                PredictionKeys.GT_TEXT: qa["answer_text"],
                PredictionKeys.MANEUVER_ID: qa["man_id"],
            }
        )

    with open(args.output_path, "w") as f:
        json.dump(all_answers, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSBench: DeepSeek Evaluation")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default="https://api.deepseek.com",
    )
    parser.add_argument(
        "--env_api_key",
        type=str,
        default="DEEPSEEK_API_KEY",
    )
    parser.add_argument(
        "--timeout", type=int, default=1, help="Timeout between API calls in seconds"
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    args = parser.parse_args()
    # infer(args)
    evaluate(args)
