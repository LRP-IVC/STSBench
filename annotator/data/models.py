import enum
from typing import List, Optional, Tuple, Union

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from shapely.geometry import MultiPoint, box

import numpy as np

from . import (
    Base,
    Position,
    ego_map_association_table,
    frame_maneuver,
    agent_maneuver,
    agent_map,
)


class LaneType(enum.Enum):
    LANE = 0
    PED_CROSSING = 1
    DRIVABLE_AREA = 2
    WALKWAY = 3


class SensorType(enum.Enum):
    LIDAR_TOP = 0
    CAM_FRONT = 1
    CAM_FRONT_RIGHT = 2
    CAM_BACK_RIGHT = 3
    CAM_BACK = 4
    CAM_BACK_LEFT = 5
    CAM_FRONT_LEFT = 6


class Map(Base):
    __tablename__ = "map"
    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str]
    lane_type: Mapped[LaneType]

    egos: Mapped[List["Ego"]] = relationship(
        secondary=ego_map_association_table, back_populates="maps"
    )

    agents: Mapped[List["Agent"]] = relationship(
        secondary=agent_map, back_populates="maps"
    )

    def __repr__(self) -> str:
        return f"Map(token={self.token!r}, lane_type={self.lane_type.name!r})"


class Ego(Base, Position):
    __tablename__ = "ego"
    id: Mapped[int] = mapped_column(primary_key=True)
    ego_pose_token: Mapped[str]
    velocity: Mapped[float | None]

    frame_id: Mapped[int] = mapped_column(ForeignKey("frame.id"))
    frame: Mapped["Frame"] = relationship(back_populates="ego")

    maps: Mapped[List[Map]] = relationship(
        secondary=ego_map_association_table, back_populates="egos"
    )

    def __repr__(self) -> str:
        return f"Ego(ego_pose_token={self.ego_pose_token!r}, x={self.x!r}, y={self.y!r}, z={self.z!r})"

    @property
    def get_lane_id(self):
        lane = [m.id for m in self.maps if m.lane_type == LaneType.LANE]
        assert len(lane) <= 1, f"There should be max one {LaneType.LANE}"
        return lane[0] if len(lane) > 0 else -1

    @property
    def velocity_kmh(self):
        return self.velocity * 3.6

    @property
    def is_human(self):
        return False

    @property
    def is_vehicle(self):
        return True


class VisibilityType(enum.Enum):
    LOW = 0
    PARTIALLY_OCCLUDED = 1
    VISIBLE = 2
    FULLY_VISIBLE = 3


class Agent(Base, Position):
    NUSCENES_NAME_MAP = {
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.wheelchair": "wheelchair",
        "human.pedestrian.stroller": "stroller",
        "human.pedestrian.personal_mobility": "personal_mobility",
        "human.pedestrian.police_officer": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "vehicle.car": "car",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.truck": "truck",
        "vehicle.construction": "construction_vehicle",
        "vehicle.emergency.ambulance": "ambulance",
        "vehicle.emergency.police": "police",
        "vehicle.trailer": "trailer",
    }

    __tablename__ = "agent"
    id: Mapped[int] = mapped_column(primary_key=True)
    sample_token: Mapped[str]
    category_name: Mapped[str]
    width: Mapped[float]
    length: Mapped[float]
    height: Mapped[float]
    vx: Mapped[float]
    vy: Mapped[float]
    vz: Mapped[float]
    visibility: Mapped[VisibilityType]

    track_id: Mapped[int] = mapped_column(ForeignKey("track.id"))
    track: Mapped["Track"] = relationship(back_populates="agents")

    frame_id: Mapped[int] = mapped_column(ForeignKey("frame.id"))
    frame: Mapped["Frame"] = relationship(back_populates="agents")

    maps: Mapped[List[Map]] = relationship(secondary=agent_map, back_populates="agents")

    maneuvers: Mapped[List["Maneuver"]] = relationship(
        secondary=agent_maneuver, back_populates="other_agents"
    )

    def __repr__(self) -> str:
        return f"Agent(sample_token={self.sample_token!r}, category_name={self.category_name!r}, x={self.x!r}, y={self.y!r}, z={self.z!r})"

    @property
    def velocity(self):
        return np.linalg.norm([self.vx, self.vy, self.vz])

    @property
    def velocity_kmh(self):
        return np.linalg.norm([self.vx, self.vy, self.vz]) * 3.6

    @property
    def lwh(self):
        return [self.length, self.width, self.height]

    @property
    def is_vehicle(self):
        return "vehicle" in self.category_name

    @property
    def is_human(self):
        return "human" in self.category_name

    @property
    def get_lane_id(self):
        lane = [m.id for m in self.maps if m.lane_type == LaneType.LANE]
        assert len(lane) <= 1, f"There should be max one {LaneType.LANE}"
        return lane[0] if len(lane) > 0 else -1

    @property
    def ped_crossing_id(self):
        lane = [m.id for m in self.maps if m.lane_type == LaneType.PED_CROSSING]
        assert len(lane) <= 1, f"There should be max one {LaneType.PED_CROSSING}"
        return lane[0] if len(lane) > 0 else -1

    @property
    def drivable_area_id(self):
        lane = [m.id for m in self.maps if m.lane_type == LaneType.DRIVABLE_AREA]
        assert len(lane) <= 1, f"There should be max one {LaneType.DRIVABLE_AREA}"
        return lane[0] if len(lane) > 0 else -1

    @property
    def walkway_id(self):
        lane = [m.id for m in self.maps if m.lane_type == LaneType.WALKWAY]
        assert len(lane) <= 1, f"There should be max one {LaneType.WALKWAY}"
        return lane[0] if len(lane) > 0 else -1

    @property
    def dist_from_ego(self):
        ego_xy = self.frame.ego.xy
        return np.linalg.norm(np.array(self.xy) - np.array(ego_xy))

    @property
    def general_class_name(self):
        return self.NUSCENES_NAME_MAP[self.category_name]

    @staticmethod
    def visibilty_from_nuscenes(nuscenes_visibility: str) -> VisibilityType:
        if nuscenes_visibility == "v0-40":
            return VisibilityType.LOW
        elif nuscenes_visibility == "v40-60":
            return VisibilityType.PARTIALLY_OCCLUDED
        elif nuscenes_visibility == "v60-80":
            return VisibilityType.VISIBLE
        elif nuscenes_visibility == "v80-100":
            return VisibilityType.FULLY_VISIBLE
        else:
            raise ValueError(f"Unknown nuScenes visibility type {nuscenes_visibility}")

    def get_position_in_lidar_frame(self) -> Position:
        lidar_sensor = self.frame.get_sensor(SensorType.LIDAR_TOP)

        xyz = self.xyz
        q = self.q

        # global -> ego
        xyz = np.array(xyz) - np.array(self.frame.ego.xyz)
        xyz = np.dot(self.frame.ego.q.inverse.rotation_matrix, xyz)
        q = self.frame.ego.q.inverse * q

        # ego -> sensor
        xyz = xyz - np.array(lidar_sensor.xyz)
        xyz = np.dot(lidar_sensor.q.inverse.rotation_matrix, xyz)
        q = lidar_sensor.q.inverse * q

        p = Position()
        p.x = xyz[0]
        p.y = xyz[1]
        p.z = xyz[2]
        p.qw = q.w
        p.qx = q.x
        p.qy = q.y
        p.qz = q.z

        return p

    def get_bbox_2d(self) -> Tuple[Tuple[float, float, float, float], SensorType]:
        for sensor_type in [
            SensorType.CAM_FRONT,
            SensorType.CAM_FRONT_RIGHT,
            SensorType.CAM_BACK_RIGHT,
            SensorType.CAM_BACK,
            SensorType.CAM_BACK_LEFT,
            SensorType.CAM_FRONT_LEFT,
        ]:
            sensor = self.frame.get_sensor(sensor_type)

            xyz = self.xyz
            q = self.q

            # global -> ego
            xyz = np.array(xyz) - np.array(self.frame.ego.xyz)
            xyz = np.dot(self.frame.ego.q.inverse.rotation_matrix, xyz)
            q = self.frame.ego.q.inverse * q

            # ego -> sensor
            xyz = xyz - np.array(sensor.xyz)
            xyz = np.dot(sensor.q.inverse.rotation_matrix, xyz)
            q = sensor.q.inverse * q

            # get corners
            w, l, h = self.width, self.length, self.height
            # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
            x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
            y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
            z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
            corners = np.vstack((x_corners, y_corners, z_corners))
            # rotate
            corners = np.dot(q.rotation_matrix, corners)
            # translate
            x, y, z = xyz
            corners[0, :] = corners[0, :] + x
            corners[1, :] = corners[1, :] + y
            corners[2, :] = corners[2, :] + z
            # Filter out the corners that are not in front of the calibrated sensor.
            in_front = np.argwhere(corners[2, :] > 0).flatten()
            corners = corners[:, in_front]

            # project to image plane
            viewpad = np.eye(4)
            viewpad[
                : sensor.camera_matrix.shape[0], : sensor.camera_matrix.shape[1]
            ] = sensor.camera_matrix
            nbr_points = corners.shape[1]
            # do operation in homogenous coordinates.
            points = np.concatenate((corners, np.ones((1, nbr_points))))
            points = np.dot(viewpad, points)
            points = points[:3, :]
            # normalize
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
            points = points.T[:, :2].tolist()

            bbox_2d = self._post_process_coords(points)
            if bbox_2d is not None:
                return bbox_2d, sensor_type
        assert False, "Object has to be visible in one camera frame."

    def _post_process_coords(
        self, corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
    ) -> Union[Tuple[float, float, float, float], None]:
        """
        Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
        intersection.
        :param corner_coords: Corner coordinates of reprojected bounding box.
        :param imsize: Size of the image canvas.
        :return: Intersection of the convex hull of the 2D box corners and the image canvas.
        """
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([
                coord for coord in img_intersection.exterior.coords
            ])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None

    def _visualize(self):
        bbox, sensor = self.get_bbox_2d()
        import cv2

        nuscenes_path = "/home/dusan/hdd/Datasets/nuscenes/v1.0-trainval/"
        img = cv2.imread(nuscenes_path + self.frame.get_sensor(sensor).path)

        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow(f"{sensor}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _visualize3d(self):
        import open3d as o3d
        from matplotlib import cm

        nuscenes_path = "/home/dusan/hdd/Datasets/nuscenes/v1.0-trainval/"
        lidar_path = nuscenes_path + self.frame.get_sensor(SensorType.LIDAR_TOP).path
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([
            -1,
            5,
        ])[:, :4]
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(points[:, :3])

        lidar_pos = self.get_position_in_lidar_frame()
        bb_3d = o3d.geometry.OrientedBoundingBox(
            lidar_pos.xyz,
            lidar_pos.q.rotation_matrix,
            [self.length, self.width, self.height],
        )

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([o3d_pcl, origin, bb_3d])


class Track(Base):
    __tablename__ = "track"
    id: Mapped[int] = mapped_column(primary_key=True)
    instance_token: Mapped[str]

    agents: Mapped[List[Agent]] = relationship(back_populates="track")

    scene_id: Mapped[int] = mapped_column(ForeignKey("scene.id"))
    scene: Mapped["Scene"] = relationship(back_populates="tracks")

    def __repr__(self) -> str:
        return f"Track(type={self.id!r}, path={self.instance_token!r})"

    def get_synchronized_agents(self, track):
        cur_agents, other_agents = [], []
        for agent in self.agents:
            for other_agent in track.agents:
                if agent.frame.id == other_agent.frame.id:
                    cur_agents.append(agent)
                    other_agents.append(other_agent)
        assert len(cur_agents) == len(other_agents)
        assert (
            np.sum(
                np.array([a.frame.timestamp for a in cur_agents])
                - np.array([a.frame.timestamp for a in other_agents])
            )
            == 0
        )
        return cur_agents, other_agents

    def get_synchronized_ego(self):
        agents = [a for a in self.agents]
        egos = [a.frame.ego for a in self.agents]
        assert (
            np.sum(
                np.array([a.frame.timestamp for a in agents])
                - np.array([e.frame.timestamp for e in egos])
            )
            == 0
        )
        return agents, egos


class Sensor(Base, Position):
    __tablename__ = "sensor"
    id: Mapped[int] = mapped_column(primary_key=True)
    sensor_token: Mapped[str]
    path: Mapped[str]
    type: Mapped[SensorType]
    fx: Mapped[Optional[float]]
    fy: Mapped[Optional[float]]
    cx: Mapped[Optional[float]]
    cy: Mapped[Optional[float]]
    width: Mapped[Optional[float]]
    height: Mapped[Optional[float]]

    frame_id: Mapped[int] = mapped_column(ForeignKey("frame.id"))
    frame: Mapped["Frame"] = relationship(back_populates="sensors")

    def __repr__(self) -> str:
        return f"Sensor(type={self.type.name!r}, path={self.path!r})"

    @property
    def is_camera(self):
        return "CAM" in self.type.name

    @property
    def is_lidar(self):
        return "LIDAR" in self.type.name

    @property
    def focal_length(self):
        assert self.is_camera
        return (self.fx, self.fy)

    @property
    def principal_point(self):
        assert self.is_camera
        return (self.cx, self.cy)

    @property
    def camera_matrix(self):
        assert self.is_camera
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


class Frame(Base):
    __tablename__ = "frame"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[int]

    scene_id: Mapped[int] = mapped_column(ForeignKey("scene.id"))
    scene: Mapped["Scene"] = relationship(back_populates="frames")

    sensors: Mapped[List[Sensor]] = relationship(back_populates="frame")
    agents: Mapped[List[Agent]] = relationship(back_populates="frame")
    ego: Mapped[Ego] = relationship(back_populates="frame")

    maneuvers: Mapped[List["Maneuver"]] = relationship(
        secondary=frame_maneuver, back_populates="frames"
    )

    def __repr__(self) -> str:
        return f"Frame(id={self.id!r}, timestamp={self.timestamp!r})"

    def get_sensor(self, sensor_type: SensorType) -> Sensor | None:
        for sensor in self.sensors:
            if sensor.type == sensor_type:
                return sensor
        return None


class Scene(Base):
    __tablename__ = "scene"
    id: Mapped[int] = mapped_column(primary_key=True)
    scene_token: Mapped[str]
    name: Mapped[str]
    location: Mapped[str]

    frames: Mapped[List[Frame]] = relationship(back_populates="scene")

    tracks: Mapped[List[Track]] = relationship(back_populates="scene")

    def __repr__(self) -> str:
        return f"Scene(token={self.scene_token!r}, name={self.name!r}, location={self.location!r})"
