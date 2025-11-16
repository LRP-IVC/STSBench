import argparse
from pathlib import Path

from .deepseek import infer
from .eval import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSBench: Llama Evaluation")
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
        default="llama3.2",
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--env_api_key",
        type=str,
        default="LLAMA_API_KEY",
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
    infer(args)
    evaluate(args)
