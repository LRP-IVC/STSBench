import argparse
import json
from pathlib import Path

# from lmdeploy import TurbomindEngineConfig, pipeline
# from lmdeploy.vl import load_image
# from lmdeploy.vl.constants import IMAGE_TOKEN
from tqdm import tqdm

from .consts import PredictionKeys
from .eval import evaluate


def main(args):
    pipe = pipeline(args.model, backend_config=TurbomindEngineConfig(tp=1))

    with open(str(args.input_path), "r") as f:
        qas = json.load(f)

    predictions = []
    for qa in tqdm(qas):
        image_paths = [str(args.nuscenes_path / c) for c in qa["cams"]]
        images = [load_image(image_path) for image_path in image_paths]
        for image in images:
            image.thumbnail((args.img_size, args.img_size))

        response = pipe((qa["prompt"].replace("{IMAGE_TOKEN}", IMAGE_TOKEN), images))

        predictions.append(
            {
                PredictionKeys.MODEL_PREDICTION: response.text,
                PredictionKeys.GT_LETTER: qa["answer_letter"],
                PredictionKeys.GT_TEXT: qa["answer_text"],
                PredictionKeys.MANEUVER_ID: qa["man_id"],
            }
        )

    with open((args.output_path), "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "OpenGVLab/InternVL2_5-1B",
            "OpenGVLab/InternVL2_5-8B",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    )
    parser.add_argument(
        "--nuscenes_path",
        type=Path,
        default=Path("/mount/data/nuscenes/v1.0-trainval/"),
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    args = parser.parse_args()

    # main(args)
    evaluate(args)
