import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from annotator.data.maneuvers import Maneuver
from annotator.data.models import Frame, SensorType, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType
from vqa_extractor.base import VQAExtractor
from vqa_extractor.senna import long_maneuver_description, short_maneuver_description


class VLMMultiImageVQAExtractor(VQAExtractor):
    def __init__(self, number_of_negatives: int = 4):
        super().__init__(
            number_of_negatives=number_of_negatives,
            ego_desc="you",
            agent_desc="object 1",
            other_agent_desc="object 2",
            long_desc_generator=long_maneuver_description,
            short_desc_generator=short_maneuver_description,
        )
        self.prompt_template.preamble = "Frame-1: {IMAGE_TOKEN}\nFrame-2: {IMAGE_TOKEN}\nFrame-3: {IMAGE_TOKEN}\nFrame-4: {IMAGE_TOKEN}\nFrame-5: {IMAGE_TOKEN}\nFrame-6: {IMAGE_TOKEN}\nYou are a helpful traffic control expert specializing in the analysis and identification of temporal actions and maneuvers performed by various agents, as well as your own temporal actions and maneuvers, in diverse driving scenarios. An agent refers to any participant in the traffic environment, including but not limited to cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles. Your task is to identify both your temporal actions and maneuvers and those of other agents within the traffic environment.\nSix cameras are strategically positioned around the vehicle to provide a 360-degree field of view. Camera data can aid in visual confirmation and potentially understanding intent. The first camera, CAM_FRONT, is positioned directly in front of the LiDAR sensor and faces forward. To the right of CAM_FRONT is CAM_FRONT_RIGHT, which is oriented at a 45-degree angle relative to the front-facing camera. On the right side of the car, CAM_BACK_RIGHT is positioned at a 135-degree angle relative to CAM_FRONT. The rear-facing camera, CAM_BACK, is oriented directly opposite to CAM_FRONT. To the left of CAM_FRONT is CAM_FRONT_LEFT, which is oriented at a -45-degree angle relative to the front-facing camera. Finally, CAM_BACK_LEFT is positioned on the left side of the car, oriented at a -135-degree angle relative to CAM_FRONT. You are provided the six sequential video frames captured at 2 frames per second."
        self.prompt_template.maneuver_description = "The following are driving maneuvers and actions along with their respective descriptions: {man_desc}"
        self.prompt_template.ego_ref = "Given that, Frame-1 is captured with {c1}, Frame-2 is captured with {c2}, Frame-3 is caputred with {c3}, Frame-4 is captured with {c4}, Frame-5 is captured with {c5}, Frame-6 is captured with {c6}, which of the following options best describes your driving maneuver?"
        self.prompt_template.ego_ref_other_agent_ref = "Consider that the Frame-1 is captured with {c1}, Frame-2 is captured with {c2}, Frame-3 is caputred with {c3}, Frame-4 is captured with {c4}, Frame-5 is captured with {c5}, Frame-6 is captured with {c6}. Also, consider {other_agent_desc}, which is {cls} inside region [{xmin}, {ymin}, {xmax}, {ymax}] in Frame-1. Which of the following options best describes your driving behavior with respect to the {other_agent_desc}?"
        self.prompt_template.agent_ref = "Consider that the Frame-1 is captured with {c1}, Frame-2 is captured with {c2}, Frame-3 is caputred with {c3}, Frame-4 is captured with {c4}, Frame-5 is captured with {c5}, Frame-6 is captured with {c6}. Also, consider {agent_desc}, which is {cls} inside region [{xmin}, {ymin}, {xmax}, {ymax}] in Frame-1. Which of the following options best describes {agent_desc} maneuver?"
        self.prompt_template.agent_ref_other_agent_ref = "Consider that the Frame-1 is captured with {c1}, Frame-2 is captured with {c2}, Frame-3 is caputred with {c3}, Frame-4 is captured with {c4}, Frame-5 is captured with {c5}, Frame-6 is captured with {c6}. Also, consider {agent_desc}, which is {cls_1} inside region [{xmin_1}, {ymin_1}, {xmax_1}, {ymax_1}] in Frame-1 and {other_agent_desc}, which is {cls_2} inside region [{xmin_2}, {ymin_2}, {xmax_2}, {ymax_2}] in Frame-1. Which of the following options best describes {agent_desc} maneuver with respect to the {other_agent_desc}?"
        self.prompt_template.postamble = "Please answer only with the letter of an option from the multiple choice list, e.g. A or B or C or D, and nothing else."

    def normalize_coordinates(self, box, image_width, image_height):
        x1, y1, x2, y2 = box
        normalized_box = [
            round((x1 / image_width) * 1000),
            round((y1 / image_height) * 1000),
            round((x2 / image_width) * 1000),
            round((y2 / image_height) * 1000),
        ]
        return normalized_box

    def generate_referal(self, man: Maneuver) -> str:
        if man.is_ego and not man.is_other_agent:
            return (
                self.prompt_template.ego_ref.format(
                    c1="CAM_FRONT",
                    c2="CAM_FRONT",
                    c3="CAM_FRONT",
                    c4="CAM_FRONT",
                    c5="CAM_FRONT",
                    c6="CAM_FRONT",
                )
                + "\n"
            )
        if man.is_ego and man.is_other_agent:
            bb2ds = [a.get_bbox_2d()[0] for a in man.other_agents]
            bb2ds = [self.normalize_coordinates(bb2d, 1600, 900) for bb2d in bb2ds]
            cams = [a.get_bbox_2d()[1] for a in man.other_agents]
            return (
                self.prompt_template.ego_ref_other_agent_ref.format(
                    c1=cams[0].name,
                    c2=cams[1].name,
                    c3=cams[2].name,
                    c4=cams[3].name,
                    c5=cams[4].name,
                    c6=cams[5].name,
                    other_agent_desc=self.other_agent_desc,
                    cls=man.other_agents[0].general_class_name,
                    xmin=int(bb2ds[0][0] / 1000.0 * 800.0),
                    ymin=int(bb2ds[0][1] / 1000.0 * 450.0),
                    xmax=int(bb2ds[0][2] / 1000.0 * 800.0),
                    ymax=int(bb2ds[0][3] / 1000.0 * 450.0),
                )
                + "\n"
            )
        if man.is_agent and not man.is_other_agent:
            bb2ds = [a.get_bbox_2d()[0] for a in man.agents]
            bb2ds = [self.normalize_coordinates(bb2d, 1600, 900) for bb2d in bb2ds]
            cams = [a.get_bbox_2d()[1] for a in man.agents]
            return (
                self.prompt_template.agent_ref.format(
                    c1=cams[0].name,
                    c2=cams[1].name,
                    c3=cams[2].name,
                    c4=cams[3].name,
                    c5=cams[4].name,
                    c6=cams[5].name,
                    agent_desc=self.agent_desc,
                    cls=man.agents[0].general_class_name,
                    xmin=int(bb2ds[0][0] / 1000.0 * 800.0),
                    ymin=int(bb2ds[0][1] / 1000.0 * 450.0),
                    xmax=int(bb2ds[0][2] / 1000.0 * 800.0),
                    ymax=int(bb2ds[0][3] / 1000.0 * 450.0),
                )
                + "\n"
            )
        if man.is_agent and man.is_other_agent:
            bb2ds_1 = [a.get_bbox_2d()[0] for a in man.agents]
            bb2ds_1 = [self.normalize_coordinates(bb2d, 1600, 900) for bb2d in bb2ds_1]
            cams = [a.get_bbox_2d()[1] for a in man.agents]

            bb2ds_2 = [a.get_bbox_2d()[0] for a in man.other_agents]
            bb2ds_2 = [self.normalize_coordinates(bb2d, 1600, 900) for bb2d in bb2ds_2]
            return (
                self.prompt_template.agent_ref_other_agent_ref.format(
                    c1=cams[0].name,
                    c2=cams[1].name,
                    c3=cams[2].name,
                    c4=cams[3].name,
                    c5=cams[4].name,
                    c6=cams[5].name,
                    agent_desc=self.agent_desc,
                    cls_1=man.agents[0].general_class_name,
                    xmin_1=int(bb2ds_1[0][0] / 1000.0 * 800.0),
                    ymin_1=int(bb2ds_1[0][1] / 1000.0 * 450.0),
                    xmax_1=int(bb2ds_1[0][2] / 1000.0 * 800.0),
                    ymax_1=int(bb2ds_1[0][3] / 1000.0 * 450.0),
                    other_agent_desc=self.other_agent_desc,
                    cls_2=man.other_agents[0].general_class_name,
                    xmin_2=int(bb2ds_2[0][0] / 1000.0 * 800.0),
                    ymin_2=int(bb2ds_2[0][1] / 1000.0 * 450.0),
                    xmax_2=int(bb2ds_2[0][2] / 1000.0 * 800.0),
                    ymax_2=int(bb2ds_2[0][3] / 1000.0 * 450.0),
                )
                + "\n"
            )
        assert False, f"Maneuver {man!r} is corrupt"


def main(args):
    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    session = Session(engine)

    stmt = select(Maneuver).where(
        Maneuver.manually_labeled == True,
        Maneuver.in_use == True,
        Maneuver.pos_maneuvers != None,
        Maneuver.pos_maneuvers.any(),
    )
    maneuvers = session.scalars(stmt).all()

    extractor = VLMMultiImageVQAExtractor()

    qas = []
    for man in tqdm(maneuvers):
        prompt, answer_text, answer_letter = extractor.generate_prompt_answers(man)

        # get images
        if man.is_ego:
            cams = [f.get_sensor(SensorType.CAM_FRONT).path for f in man.frames]
        elif man.is_agent:
            cam_types = [a.get_bbox_2d()[1] for a in man.agents]
            cams = [f.get_sensor(c).path for f, c in zip(man.frames, cam_types)]
        else:
            assert False

        qa = dict(
            prompt=prompt,
            answer_text=answer_text,
            answer_letter=answer_letter,
            man_id=man.id,
            cams=cams,
        )
        qas.append(qa)

    with open(args.save_path, "w") as f:
        json.dump(qas, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSBench Annotator")
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
    )
    args = parser.parse_args()

    main(args)
