import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from vqa_extractor.base import VQAExtractor
from annotator.data.maneuvers import Maneuver
from annotator.data.models import Frame, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType


class LanguageOnlyVQAExtractor(VQAExtractor):
    def __init__(self, number_of_negatives: int = 4):
        super().__init__(
            number_of_negatives=number_of_negatives,
            ego_desc="ego",
            agent_desc="agent 1",
            other_agent_desc="agent 2",
        )

        self.prompt_template.preamble = """You are a helpful traffic control expert specializing in the analysis and identification of temporal actions and maneuvers performed by various agents in diverse driving scenarios.|An agent refers to any participant in the traffic environment, including but not limited to cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles. The traffic environment is captured using a sophisticated suite of sensors mounted on an ego vehicle, which is a standard passenger car.
The sensor suite includes a Light Detection and Ranging (LiDAR) sensor labeled LIDAR_TOP, mounted on the roof of the car. This LiDAR sensor provides high-precision 3D spatial data about the surrounding environment. LiDAR data is crucial for precise spatial positioning and size, which helps in differentiating vehicle types and detecting movement. For each agent, the LiDAR data includes the frame number, the center position of the agent (x, y, z) in meters relative to the LiDAR, and a heading angle in degrees relative to the LiDAR, and size dimensions such as width, length, and height in meters.
In addition to the LiDAR, six cameras are strategically positioned around the vehicle to provide a 360-degree field of view. Camera data can aid in visual confirmation and potentially understanding intent. The first camera, CAM_FRONT, is positioned directly in front of the LiDAR sensor and faces forward. To the right of CAM_FRONT is CAM_FRONT_RIGHT, which is oriented at a 45-degree angle relative to the front-facing camera. On the right side of the car, CAM_BACK_RIGHT is positioned at a 135-degree angle relative to CAM_FRONT. The rear-facing camera, CAM_BACK, is oriented directly opposite to CAM_FRONT. To the left of CAM_FRONT is CAM_FRONT_LEFT, which is oriented at a -45-degree angle relative to the front-facing camera. Finally, CAM_BACK_LEFT is positioned on the left side of the car, oriented at a -135-degree angle relative to CAM_FRONT. For each agent, the camera data includes the frame number, the center of the agent given with pixel location (x, y) in the image frame and in which camera is agent visible (e.g. CAM_FRONT or CAM_BACK).
The system also integrates a GPS sensor, which provides the ego vehicle's precise global position. The GPS data includes the frame number, coordinates (x, y) in meters within a global coordinate system and the vehicle's orientation in radians relative to the global frame.
Together, this comprehensive sensor suite enables detailed monitoring and analysis of the dynamic behaviors of all traffic agents. Your task is to leverage this data to identify and interpret the temporal actions and maneuvers of each agent within the traffic environment."""
        self.prompt_template.maneuver_description = "The following are driving maneuvers and actions along with their respective descriptions: {man_desc}"
        self.prompt_template.ego_ref = """Your input consists of sequential data, captured over {n_frames} frames and {secs} seconds with the described sensor suite.
Ego:
{ego_data}
Which of the following options best describes ego driving maneuver?"""
        self.prompt_template.ego_ref_other_agent_ref = """Your input consists of sequential data, captured over {n_frames} frames and {secs} seconds with the described sensor suite.
Ego:
{ego_data}
Agent 2:
{agent1_data}
Which of the following options best describes ego driving behavior with respect to agent 2?"""
        self.prompt_template.agent_ref = """Your input consists of sequential data, captured over {n_frames} frames and {secs} seconds with the described sensor suite.
Ego:
{ego_data}
Agent 1:
{agent1_data}
Which of the following options best describes agent 1 driving maneuver?"""
        self.prompt_template.agent_ref_other_agent_ref = """Your input consists of sequential data, captured over {n_frames} frames and {secs} seconds with the described sensor suite.
Ego:
{ego_data}
Agent 1:
{agent1_data}
Agent 2:
{agent2_data}
Which of the following options best describes agent 1 driving behaviour with respect to agent 2?"""
        self.prompt_template.postamble = "Please answer only with the letter of an option from the multiple choice list, e.g. A or B or C or D, and nothing else."

    def generate_referal(self, man: Maneuver) -> str:
        if man.is_ego and not man.is_other_agent:
            pos = [e for e in man.egos]
            xyzs = [
                p.translate_rotate_xyz(-np.array(pos[0].xyz), pos[0].q.inverse)
                for p in pos
            ]
            rots = [p.rotate_yaw(pos[0].q.inverse) for p in pos]

            ego_data = ""
            for i, (xyz, rot) in enumerate(zip(xyzs, rots)):
                ego_data += f"  Frame number: {i}\n  x: {xyz[0]:.2f}\n  y: {xyz[1]:.2f}\n  rotation: {rot:.2f}\n\n"

            return (
                self.prompt_template.ego_ref.format(
                    n_frames=len(man.egos), secs=6, ego_data=ego_data
                )
                + "\n"
            )
        if man.is_ego and man.is_other_agent:
            pos = [e for e in man.egos]
            xyzs = [
                p.translate_rotate_xyz(-np.array(pos[0].xyz), pos[0].q.inverse)
                for p in pos
            ]
            rots = [p.rotate_yaw(pos[0].q.inverse) for p in pos]

            ego_data = ""
            for i, (xyz, rot) in enumerate(zip(xyzs, rots)):
                ego_data += f"  Frame number: {i}\n  x: {xyz[0]:.2f}\n  y: {xyz[1]:.2f}\n  rotation: {rot:.2f}\n\n"

            agent_pos_lidar = [
                a.get_position_in_lidar_frame() for a in man.other_agents
            ]
            agent_cam = [a.get_bbox_2d() for a in man.other_agents]
            agent1_data = ""
            for i, (lid_pos, cam) in enumerate(zip(agent_pos_lidar, agent_cam)):
                agent1_data += f"  Frame number: {i}\n  LiDAR x: {lid_pos.x:.2f}\n  LiDAR y: {lid_pos.y:.2f}\n  LiDAR rotation: {lid_pos.yaw:.2f}\n"
                agent1_data += f"  CAM x: {(cam[0][2] + cam[0][0]) / 2.0:.2f}\n  CAM y: {(cam[0][3] + cam[0][1]) / 2.0:.2f}\n  CAM: {cam[1].name}\n\n"
            return (
                self.prompt_template.ego_ref_other_agent_ref.format(
                    n_frames=len(man.egos),
                    secs=6,
                    ego_data=ego_data,
                    agent1_data=agent1_data,
                )
                + "\n"
            )
        if man.is_agent and not man.is_other_agent:
            pos = [e for e in man.egos]
            xyzs = [
                p.translate_rotate_xyz(-np.array(pos[0].xyz), pos[0].q.inverse)
                for p in pos
            ]
            rots = [p.rotate_yaw(pos[0].q.inverse) for p in pos]

            ego_data = ""
            for i, (xyz, rot) in enumerate(zip(xyzs, rots)):
                ego_data += f"  Frame number: {i}\n  x: {xyz[0]:.2f}\n  y: {xyz[1]:.2f}\n  rotation: {rot:.2f}\n\n"

            agent_pos_lidar = [a.get_position_in_lidar_frame() for a in man.agents]
            agent_cam = [a.get_bbox_2d() for a in man.agents]
            agent1_data = f"  Class: {man.agents[0].general_class_name}\n\n"
            for i, (lid_pos, cam) in enumerate(zip(agent_pos_lidar, agent_cam)):
                agent1_data += f"  Frame number: {i}\n  LiDAR x: {lid_pos.x:.2f}\n  LiDAR y: {lid_pos.y:.2f}\n  LiDAR rotation: {lid_pos.yaw:.2f}\n"
                agent1_data += f"  CAM x: {(cam[0][2] + cam[0][0]) / 2.0:.2f}\n  CAM y: {(cam[0][3] + cam[0][1]) / 2.0:.2f}\n  CAM: {cam[1].name}\n\n"
            return (
                self.prompt_template.agent_ref.format(
                    n_frames=len(man.egos),
                    secs=6,
                    ego_data=ego_data,
                    agent1_data=agent1_data,
                )
                + "\n"
            )
        if man.is_agent and man.is_other_agent:
            pos = [e for e in man.egos]
            xyzs = [
                p.translate_rotate_xyz(-np.array(pos[0].xyz), pos[0].q.inverse)
                for p in pos
            ]
            rots = [p.rotate_yaw(pos[0].q.inverse) for p in pos]

            ego_data = ""
            for i, (xyz, rot) in enumerate(zip(xyzs, rots)):
                ego_data += f"  Frame number: {i}\n  x: {xyz[0]:.2f}\n  y: {xyz[1]:.2f}\n  rotation: {rot:.2f}\n\n"

            agent_pos_lidar = [a.get_position_in_lidar_frame() for a in man.agents]
            agent_cam = [a.get_bbox_2d() for a in man.agents]
            agent1_data = f"  Class: {man.agents[0].general_class_name}\n\n"
            for i, (lid_pos, cam) in enumerate(zip(agent_pos_lidar, agent_cam)):
                agent1_data += f"  Frame number: {i}\n  LiDAR x: {lid_pos.x:.2f}\n  LiDAR y: {lid_pos.y:.2f}\n  LiDAR rotation: {lid_pos.yaw:.2f}\n"
                agent1_data += f"  CAM x: {(cam[0][2] + cam[0][0]) / 2.0:.2f}\n  CAM y: {(cam[0][3] + cam[0][1]) / 2.0:.2f}\n  CAM: {cam[1].name}\n\n"

            agent_pos_lidar = [
                a.get_position_in_lidar_frame() for a in man.other_agents
            ]
            agent_cam = [a.get_bbox_2d() for a in man.other_agents]
            agent2_data = f"  Class: {man.other_agents[0].general_class_name}\n\n"
            for i, (lid_pos, cam) in enumerate(zip(agent_pos_lidar, agent_cam)):
                agent2_data += f"  Frame number: {i}\n  LiDAR x: {lid_pos.x:.2f}\n  LiDAR y: {lid_pos.y:.2f}\n  LiDAR rotation: {lid_pos.yaw:.2f}\n"
                agent2_data += f"  CAM x: {(cam[0][2] + cam[0][0]) / 2.0:.2f}\n  CAM y: {(cam[0][3] + cam[0][1]) / 2.0:.2f}\n  CAM: {cam[1].name}\n\n"

            return (
                self.prompt_template.agent_ref_other_agent_ref.format(
                    n_frames=len(man.egos),
                    secs=6,
                    ego_data=ego_data,
                    agent1_data=agent1_data,
                    agent2_data=agent2_data,
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

    extractor = LanguageOnlyVQAExtractor()

    qas = []
    for man in tqdm(maneuvers):
        prompt, answer_text, answer_letter = extractor.generate_prompt_answers(man)
        qa = dict(
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
