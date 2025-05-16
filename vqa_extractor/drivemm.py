import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from annotator.data.maneuvers import Maneuver
from annotator.data.models import Frame, SensorType, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType
from vqa_extractor.base import VQAExtractor


class OmniDriveVQAExtractor(VQAExtractor):
    def __init__(self, number_of_negatives: int = 4):
        super().__init__(
            number_of_negatives=number_of_negatives,
            ego_desc="ego",
            agent_desc="c1",
            other_agent_desc="c2",
        )

        self.prompt_template.preamble = "1: <video> 2: <video> 3: <video> 4: <video> 5: <video> 6: <video>. These six images are the front view, front left view, front right view, back view, back left view and back right view of the ego vehicle. You are a helpful traffic control expert specializing in analyzing and identifying the temporal actions and maneuvers of the ego vehicle and other agents in diverse driving scenarios. Agents include all traffic participants such as cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles. Your task is to identify the temporal actions and maneuvers of both ego and other agents within the traffic environment."
        self.prompt_template.maneuver_description = "The following are driving maneuvers and actions along with their respective descriptions: {man_desc}"
        self.prompt_template.ego_ref = (
            "Which of the following options best describes ego driving maneuver?"
        )
        self.prompt_template.ego_ref_other_agent_ref = "Which of the following options best describes the ego driving behavior with respect to the <{other_agent_desc},{cam},{x},{y}>?"
        self.prompt_template.agent_ref = "Which of the following options best describes the driving behavior of the <{agent_desc},{cam},{x},{y}>?"
        self.prompt_template.agent_ref_other_agent_ref = "Which of the following options best describes <{agent_desc},{cam_1},{x_1},{y_1}> maneuver with respect to the <{other_agent_desc},{cam_2},{x_2},{y_2}>?"
        self.prompt_template.postamble = "Please answer only with the letter of an option from the multiple choice list, e.g. A or B or C or D, and nothing else."

    def normalize_coordinates_center(self, box, image_width, image_height):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        normalized_box = [
            round((cx / image_width) * 100),
            round((cy / image_height) * 100),
        ]
        return normalized_box

    def generate_prompt_answers(self, man: Maneuver):
        prompt, answer_text, answer_letter = super().generate_prompt_answers(man)
        prompt = prompt.replace("C1", "c1")
        prompt = prompt.replace("C2", "c2")
        return prompt, answer_text, answer_letter

    def generate_referal(self, man: Maneuver) -> str:
        if man.is_ego and not man.is_other_agent:
            return self.prompt_template.ego_ref.format() + "\n"
        if man.is_ego and man.is_other_agent:
            bb2d, cam = man.other_agents[0].get_bbox_2d()
            xy = self.normalize_coordinates_center(bb2d, 1600, 900)
            return (
                self.prompt_template.ego_ref_other_agent_ref.format(
                    other_agent_desc=self.other_agent_desc,
                    cam=cam.name,
                    x=xy[0],
                    y=xy[1],
                )
                + "\n"
            )
        if man.is_agent and not man.is_other_agent:
            bb2d, cam = man.agents[0].get_bbox_2d()
            xy = self.normalize_coordinates_center(bb2d, 1600, 900)
            return (
                self.prompt_template.agent_ref.format(
                    agent_desc=self.agent_desc, cam=cam.name, x=xy[0], y=xy[1]
                )
                + "\n"
            )
        if man.is_agent and man.is_other_agent:
            bb2d_1, cam_1 = man.agents[0].get_bbox_2d()
            xy_1 = self.normalize_coordinates_center(bb2d_1, 1600, 900)
            bb2d_2, cam_2 = man.agents[0].get_bbox_2d()
            xy_2 = self.normalize_coordinates_center(bb2d_2, 1600, 900)
            return (
                self.prompt_template.agent_ref_other_agent_ref.format(
                    agent_desc=self.agent_desc,
                    cam_1=cam_1.name,
                    x_1=xy_1[0],
                    y_1=xy_1[1],
                    other_agent_desc=self.other_agent_desc,
                    cam_2=cam_2.name,
                    x_2=xy_2[0],
                    y_2=xy_2[1],
                )
                + "\n"
            )
        assert False, f"Maneuver {man!r} is corrupt"


def remove_our_existing_questions(out):
    # This will remove STS questions from all json files in OmniDrive vqa folder
    for od_qa_path in out.iterdir():
        if not od_qa_path.is_file():
            continue

        with open(od_qa_path, "r") as f:
            od_qas = json.load(f)

        # remove previos questions from us if exist
        keep_od_qa = []
        for od_qa in od_qas:
            if not od_qa.get("STS_BENCH", False):
                keep_od_qa.append(od_qa)

        with open(od_qa_path, "w") as json_file:
            json.dump(keep_od_qa, json_file, indent=4)
            diff = abs(len(od_qas) - len(keep_od_qa))
            if diff > 0:
                print(f"Removed {diff} questions from {od_qa_path.stem}")


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

    extractor = OmniDriveVQAExtractor()

    cam_order = [
        SensorType.CAM_FRONT,
        SensorType.CAM_FRONT_RIGHT,
        SensorType.CAM_FRONT_LEFT,
        SensorType.CAM_BACK,
        SensorType.CAM_BACK_LEFT,
        SensorType.CAM_BACK_RIGHT,
    ]
    qas = []
    for man in tqdm(maneuvers):
        prompt, answer_text, answer_letter = extractor.generate_prompt_answers(man)

        sensors = {}
        # add all sensor paths
        for sensor_type in cam_order:
            for frame in man.frames:
                sensor = frame.get_sensor(sensor_type)
                key = f"{sensor_type.name}"
                if key not in sensors:
                    sensors[key] = []
                sensors[key].append(sensor.path)

        qa = dict(
            images=sensors,
            prompt=prompt,
            answer_text=answer_text,
            answer_letter=answer_letter,
            man_id=man.id,
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
