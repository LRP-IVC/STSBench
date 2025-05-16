import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import numpy as np
from nuscenes.nuscenes import NuScenes
import pandas as pd
from tqdm import tqdm

from annotator.mining.consts import short_maneuver_description
from vqa_extractor.base import VQAExtractor
from annotator.data.maneuvers import Maneuver
from annotator.data.models import Frame, SensorType, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType


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

    all_data = []
    for man in tqdm(maneuvers):
        man_data = {}

        man_data["scenario"] = short_maneuver_description(
            man.pos_maneuvers[0].type, man.is_ego, man.is_agent
        )
        man_data["negative_scenarios"] = ",".join([
            short_maneuver_description(n.type, man.is_ego, man.is_agent)
            for n in man.neg_maneuvers
        ])

        sensors = {}
        # add all sensor paths
        for sensor_type in SensorType:
            for frame in man.frames:
                sensor = frame.get_sensor(sensor_type)
                key = f"{sensor_type.name}"
                if key not in sensors:
                    sensors[key] = []
                sensors[key].append(sensor.path)
        man_data["sensors"] = sensors

        # add ego data
        if man.is_ego:
            man_data["ego"] = {
                "xyz": [[fr.ego.x, fr.ego.y, fr.ego.z] for fr in man.frames],
                "yaw": [fr.ego.yaw for fr in man.frames],
                "velocity": [fr.ego.velocity for fr in man.frames],
                "ego_pose_token": [fr.ego.ego_pose_token for fr in man.frames],
            }

        # add agent data
        if man.is_agent:
            man_data["agent"] = {
                "global": {
                    "xyz": [[a.x, a.y, a.z] for a in man.agents],
                    "yaw": [a.yaw for a in man.agents],
                },
                "lidar": {
                    "xyz": [
                        [
                            a.get_position_in_lidar_frame().x,
                            a.get_position_in_lidar_frame().y,
                            a.get_position_in_lidar_frame().z,
                        ]
                        for a in man.agents
                    ],
                    "wlh": [
                        [
                            a.width,
                            a.length,
                            a.height,
                        ]
                        for a in man.agents
                    ],
                    "yaw": [a.get_position_in_lidar_frame().yaw for a in man.agents],
                    "velocity": [a.velocity for a in man.agents],
                },
                "camera": {
                    "bb2d": [
                        [
                            a.get_bbox_2d()[0][0],
                            a.get_bbox_2d()[0][1],
                            a.get_bbox_2d()[0][2],
                            a.get_bbox_2d()[0][3],
                        ]
                        for a in man.agents
                    ],
                    "cam": [a.get_bbox_2d()[1].name for a in man.agents],
                },
                "agent_token": [a.sample_token for a in man.agents],
            }

        # add other agent data
        if man.is_other_agent:
            man_data["other_agent"] = {
                "global": {
                    "xyz": [[a.x, a.y, a.z] for a in man.other_agents],
                    "yaw": [a.yaw for a in man.other_agents],
                },
                "lidar": {
                    "xyz": [
                        [
                            a.get_position_in_lidar_frame().x,
                            a.get_position_in_lidar_frame().y,
                            a.get_position_in_lidar_frame().z,
                        ]
                        for a in man.other_agents
                    ],
                    "wlh": [
                        [
                            a.width,
                            a.length,
                            a.height,
                        ]
                        for a in man.other_agents
                    ],
                    "yaw": [
                        a.get_position_in_lidar_frame().yaw for a in man.other_agents
                    ],
                    "velocity": [a.velocity for a in man.other_agents],
                },
                "camera": {
                    "bb2d": [
                        [
                            a.get_bbox_2d()[0][0],
                            a.get_bbox_2d()[0][1],
                            a.get_bbox_2d()[0][2],
                            a.get_bbox_2d()[0][3],
                        ]
                        for a in man.other_agents
                    ],
                    "cam": [a.get_bbox_2d()[1].name for a in man.other_agents],
                },
                "agent_token": [a.sample_token for a in man.other_agents],
            }
        all_data.append(man_data)

    with open(args.save_path, "w") as f:
        json.dump(all_data, f, indent=4)


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
