import argparse
from pathlib import Path
from functools import partial
import math

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.map_expansion.map_api import NuScenesMap
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session
from tqdm import tqdm

from annotator.data.models import (
    Base,
    Scene,
    Frame,
    Ego,
    Map,
    LaneType,
    Agent,
    Track,
    Sensor,
    SensorType,
)
from annotator.data.maneuvers import Maneuver
import annotator.data.utils as adu


def map_extractor(nusc_map, session):
    from nuscenes.map_expansion.map_api import locations

    all_db_maps = []
    for location in locations:
        curr_nusc_map = nusc_map(map_name=location)
        for lane in curr_nusc_map.lane:
            db_map = Map(
                token=lane["token"],
                lane_type=LaneType.LANE,
            )
            all_db_maps.append(db_map)

        for ped_crossing in curr_nusc_map.ped_crossing:
            db_map = Map(
                token=ped_crossing["token"],
                lane_type=LaneType.PED_CROSSING,
            )
            all_db_maps.append(db_map)

        for drivable_area in curr_nusc_map.drivable_area:
            db_map = Map(
                token=drivable_area["token"],
                lane_type=LaneType.DRIVABLE_AREA,
            )
            all_db_maps.append(db_map)

        for walkway in curr_nusc_map.walkway:
            db_map = Map(
                token=walkway["token"],
                lane_type=LaneType.WALKWAY,
            )
            all_db_maps.append(db_map)

    session.add_all(all_db_maps)
    session.commit()


def scene_extractor(nusc, nusc_map, scenes, session):
    pbar = tqdm(total=len(scenes))

    for scene in nusc.scene:
        if scene["name"] not in scenes:
            continue

        pbar.update(1)

        map_name = nusc.get("log", scene["log_token"])["location"]
        curr_nusc_map = nusc_map(map_name=map_name)

        db_scene = Scene(
            scene_token=scene["token"], name=scene["name"], location=map_name
        )

        cur_sample_token = scene["first_sample_token"]
        last_sample_token = scene["last_sample_token"]
        while True:
            sample = nusc.get("sample", cur_sample_token)

            db_frame = Frame(timestamp=sample["timestamp"], scene=db_scene)

            # add sensors to the current frame
            db_sensors = []
            for sensor in SensorType:
                sensor_data = nusc.get("sample_data", sample["data"][sensor.name])
                calibrated_sensor_token = sensor_data["calibrated_sensor_token"]
                calibrated_sensor = nusc.get(
                    "calibrated_sensor", calibrated_sensor_token
                )
                db_sensors.append(
                    Sensor(
                        sensor_token=sensor_data["token"],
                        path=sensor_data["filename"],
                        type=sensor,
                        x=calibrated_sensor["translation"][0],
                        y=calibrated_sensor["translation"][1],
                        z=calibrated_sensor["translation"][2],
                        qw=calibrated_sensor["rotation"][0],
                        qx=calibrated_sensor["rotation"][1],
                        qy=calibrated_sensor["rotation"][2],
                        qz=calibrated_sensor["rotation"][3],
                        frame=db_frame,
                    )
                )
                if len(calibrated_sensor["camera_intrinsic"]) > 0:
                    db_sensors[-1].fx = calibrated_sensor["camera_intrinsic"][0][0]
                    db_sensors[-1].fy = calibrated_sensor["camera_intrinsic"][1][1]
                    db_sensors[-1].cx = calibrated_sensor["camera_intrinsic"][0][2]
                    db_sensors[-1].cy = calibrated_sensor["camera_intrinsic"][1][2]
                    db_sensors[-1].height = sensor_data["height"]
                    db_sensors[-1].width = sensor_data["width"]

            session.add_all(db_sensors)

            # add ego
            sensor_data = nusc.get(
                "sample_data", sample["data"][SensorType.LIDAR_TOP.name]
            )
            ego_pose = nusc.get("ego_pose", sensor_data["ego_pose_token"])

            db_ego = Ego(
                ego_pose_token=ego_pose["token"],
                x=ego_pose["translation"][0],
                y=ego_pose["translation"][1],
                z=ego_pose["translation"][2],
                qw=ego_pose["rotation"][0],
                qx=ego_pose["rotation"][1],
                qy=ego_pose["rotation"][2],
                qz=ego_pose["rotation"][3],
                frame=db_frame,
            )
            session.add(db_ego)

            ego_map_layers = curr_nusc_map.layers_on_point(
                ego_pose["translation"][0], ego_pose["translation"][1], ["lane"]
            )
            if ego_map_layers["lane"] != "":
                stmt = select(Map).where(Map.token == ego_map_layers["lane"])
                db_map = session.scalars(stmt).one()
                db_ego.maps.append(db_map)

            # add agents
            for sample_annotation_token in sample["anns"]:
                sample_annotation = nusc.get(
                    "sample_annotation", sample_annotation_token
                )

                category_name = sample_annotation["category_name"]
                if "vehicle" not in category_name and "human" not in category_name:
                    continue

                box = nusc.get_box(sample_annotation["token"])
                box.velocity = nusc.box_velocity(box.token)

                instance_token = sample_annotation["instance_token"]

                stmt = select(Track).where(Track.instance_token == instance_token)
                db_track = session.scalars(stmt).one_or_none()
                if db_track is None:
                    db_track = Track(instance_token=instance_token, scene=db_scene)
                    session.add(db_track)

                db_agent = Agent(
                    sample_token=sample_annotation_token,
                    category_name=category_name,
                    x=box.center[0],
                    y=box.center[1],
                    z=box.center[2],
                    qw=box.orientation[0],
                    qx=box.orientation[1],
                    qy=box.orientation[2],
                    qz=box.orientation[3],
                    width=box.wlh[0],
                    length=box.wlh[1],
                    height=box.wlh[2],
                    vx=0 if math.isnan(box.velocity[0]) else box.velocity[0],
                    vy=0 if math.isnan(box.velocity[1]) else box.velocity[1],
                    vz=0 if math.isnan(box.velocity[2]) else box.velocity[2],
                    track=db_track,
                    frame=db_frame,
                    visibility=Agent.visibilty_from_nuscenes(
                        nusc.get("visibility", sample_annotation["visibility_token"])[
                            "level"
                        ]
                    ),
                )
                session.add(db_agent)

                # add maps
                map_layers = curr_nusc_map.layers_on_point(
                    box.center[0],
                    box.center[1],
                    ["lane", "ped_crossing", "drivable_area", "walkway"],
                )
                if map_layers["lane"] != "":
                    stmt = select(Map).where(Map.token == map_layers["lane"])
                    db_map = session.scalars(stmt).one()
                    db_agent.maps.append(db_map)

                if map_layers["ped_crossing"] != "":
                    stmt = select(Map).where(Map.token == map_layers["ped_crossing"])
                    db_map = session.scalars(stmt).one()
                    db_agent.maps.append(db_map)

                if map_layers["drivable_area"] != "":
                    stmt = select(Map).where(Map.token == map_layers["drivable_area"])
                    db_map = session.scalars(stmt).one()
                    db_agent.maps.append(db_map)

                if map_layers["walkway"] != "":
                    stmt = select(Map).where(Map.token == map_layers["walkway"])
                    db_map = session.scalars(stmt).one()
                    db_agent.maps.append(db_map)

            session.commit()

            cur_sample_token = sample["next"]
            if cur_sample_token == last_sample_token:
                break


def compute_ego_vel(session):
    stmt = select(Scene)
    scenes = session.scalars(stmt).all()
    for scene in scenes:
        vel = adu.compute_vel([f.ego for f in scene.frames])
        for i in range(len(vel)):
            scene.frames[i].ego.velocity = vel[i]
        session.commit()


def main():
    parser = argparse.ArgumentParser(
        prog="nuScenes ORM extractor", usage="%(prog)s [options]"
    )
    parser.add_argument(
        "--dataroot",
        help="nuScenes data root folder",
        default=Path("./nuscenes/v1.0-trainval"),
        type=Path,
    )
    parser.add_argument(
        "--version",
        help="nuScenes data version",
        default="trainval",
        choices=["trainval", "mini"],
    )
    parser.add_argument(
        "--split",
        help="nuScenes data split",
        default="val",
        choices=["val"],
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    args = parser.parse_args()

    nusc = NuScenes(
        version=f"v1.0-{args.version}", dataroot=args.dataroot, verbose=False
    )
    nusc_map = partial(NuScenesMap, dataroot=args.dataroot)
    scenes = getattr(splits, args.split)

    # Warning: existing db will be removed
    args.db_path.unlink(missing_ok=True)

    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    Base.metadata.create_all(engine)
    session = Session(engine)

    map_extractor(nusc_map, session)
    scene_extractor(nusc, nusc_map, scenes, session)
    compute_ego_vel(session)


if __name__ == "__main__":
    main()
