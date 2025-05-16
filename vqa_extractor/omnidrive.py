import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from annotator.data.maneuvers import Maneuver
from annotator.data.models import Frame, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType
from vqa_extractor.base import VQAExtractor
from vqa_extractor.senna import long_maneuver_description, short_maneuver_description


class OmniDriveVQAExtractor(VQAExtractor):
    def __init__(self, number_of_negatives: int = 4):
        super().__init__(
            number_of_negatives=number_of_negatives,
            ego_desc="you",
            agent_desc="object 1",
            other_agent_desc="object 2",
            long_desc_generator=long_maneuver_description,
            short_desc_generator=short_maneuver_description,
        )

        self.prompt_template.preamble = "You are a helpful traffic control expert specializing in the analysis and identification of temporal actions and maneuvers performed by various agents, as well as your own temporal actions and maneuvers, in diverse driving scenarios. An agent refers to any participant in the traffic environment, including but not limited to cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles. Your task is to identify both your temporal actions and maneuvers and those of other agents within the traffic environment."
        self.prompt_template.maneuver_description = "The following are driving maneuvers and actions along with their respective descriptions: {man_desc}"
        self.prompt_template.ego_ref = (
            "Which of the following options best describes your driving maneuver?"
        )
        self.prompt_template.ego_ref_other_agent_ref = "Consider the {other_agent_desc}, which is a {cls} located at coordinates ({x:+.1f}, {y:+.1f}) and moving at a velocity of {vel:.1f} m/s. Which of the following options best describes your driving behavior with respect to the {other_agent_desc}?"
        self.prompt_template.agent_ref = "Consider the {agent_desc}, which is a {cls} located at coordinates ({x:+.1f}, {y:+.1f}) and moving at a velocity of {vel:.1f} m/s. Which of the following options best describes {agent_desc} maneuver?"
        self.prompt_template.agent_ref_other_agent_ref = "Consider the {agent_desc}, which is a {cls_1} located at coordinates ({x_1:+.1f}, {y_1:+.1f}) and moving at a velocity of {vel_1:.1f} m/s, and the {other_agent_desc}, whic is a {cls_2} located at coordinates ({x_2:+.1f}, {y_2:+.1f}) and moving at a velocity of {vel_2:.1f} m/s. Which of the following options best describes {agent_desc} maneuver with respect to the {other_agent_desc}?"
        self.prompt_template.postamble = "Please answer only with the letter of an option from the multiple choice list, e.g. A or B or C or D, and nothing else."

    def generate_referal(self, man: Maneuver) -> str:
        if man.is_ego and not man.is_other_agent:
            return self.prompt_template.ego_ref.format() + "\n"
        if man.is_ego and man.is_other_agent:
            return (
                self.prompt_template.ego_ref_other_agent_ref.format(
                    cls=man.other_agents[0].general_class_name,
                    x=man.other_agents[0].get_position_in_lidar_frame().x,
                    y=man.other_agents[0].get_position_in_lidar_frame().y,
                    vel=man.other_agents[0].velocity,
                    other_agent_desc=self.other_agent_desc,
                )
                + "\n"
            )
        if man.is_agent and not man.is_other_agent:
            return (
                self.prompt_template.agent_ref.format(
                    cls=man.agents[0].general_class_name,
                    x=man.agents[0].get_position_in_lidar_frame().x,
                    y=man.agents[0].get_position_in_lidar_frame().y,
                    vel=man.agents[0].velocity,
                    agent_desc=self.agent_desc,
                )
                + "\n"
            )
        if man.is_agent and man.is_other_agent:
            return (
                self.prompt_template.agent_ref_other_agent_ref.format(
                    agent_desc=self.agent_desc,
                    cls_1=man.agents[0].general_class_name,
                    x_1=man.agents[0].get_position_in_lidar_frame().x,
                    y_1=man.agents[0].get_position_in_lidar_frame().y,
                    vel_1=man.agents[0].velocity,
                    other_agent_desc=self.other_agent_desc,
                    cls_2=man.other_agents[0].general_class_name,
                    x_2=man.other_agents[0].get_position_in_lidar_frame().x,
                    y_2=man.other_agents[0].get_position_in_lidar_frame().y,
                    vel_2=man.other_agents[0].velocity,
                )
                + "\n"
            )
        assert False, f"Maneuver {man!r} is corrupt"


def main(args):
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=True)

    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    session = Session(engine)

    stmt = select(Maneuver).where(
        Maneuver.manually_labeled == True,
        Maneuver.in_use == True,
        Maneuver.pos_maneuvers != None,
        Maneuver.pos_maneuvers.any(),
    )
    maneuvers = session.scalars(stmt).all()

    extractor = OmniDriveVQAExtractor()

    qas = []
    for man in tqdm(maneuvers):
        prompt, answer_text, answer_letter = extractor.generate_prompt_answers(man)

        # get sample_token
        sample_token = nusc.get("sample_data", man.frames[0].sensors[0].sensor_token)[
            "sample_token"
        ]

        qa = {
            "question": prompt,
            "answer_text": answer_text,
            "answer_letter": answer_letter,
            "man_id": man.id,
        }
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
        "--dataroot",
        help="nuScenes data root folder",
        default=Path("./nuscenes/v1.0-trainval"),
        type=Path,
    )
    parser.add_argument(
        "--save_path",
        type=Path,
    )
    args = parser.parse_args()

    main(args)
