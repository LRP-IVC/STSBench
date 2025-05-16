from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import DeclarativeBase
from pyquaternion import Quaternion
import numpy as np
import numpy.typing as npt


class Base(DeclarativeBase):
    pass


class Position:
    x: Mapped[float]
    y: Mapped[float]
    z: Mapped[float]
    qw: Mapped[float]
    qx: Mapped[float]
    qy: Mapped[float]
    qz: Mapped[float]

    @property
    def xyz(self):
        return [self.x, self.y, self.z]

    @property
    def xy(self):
        return [self.x, self.y]

    @property
    def qxyzw(self):
        return [self.qx, self.qy, self.qz, self.qw]

    @property
    def qwxyz(self):
        return [self.qw, self.qx, self.qy, self.qz]

    @property
    def q(self):
        return Quaternion(self.qwxyz)

    @property
    def yaw(self):
        return self.quaternion_yaw(self.q)

    def quaternion_yaw(self, q: Quaternion) -> float:
        """
        Calculate the yaw angle from a quaternion.
        Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
        It does not work for a box in the camera frame.
        :param q: Quaternion of interest.
        :return: Yaw angle in radians.
        """

        # Project into xy plane.
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

        # Measure yaw using arctan.
        yaw = np.arctan2(v[1], v[0])

        return yaw

    def translate_rotate_xyz(self, translation: npt.ArrayLike, rotation: Quaternion):
        xyz = np.array(self.xyz) + translation
        xyz = np.dot(rotation.rotation_matrix, xyz)
        return xyz

    def rotate_yaw(self, rotation: Quaternion):
        return self.quaternion_yaw(rotation * self.q)


ego_map_association_table = Table(
    "ego_map_association_table",
    Base.metadata,
    Column("ego_id", ForeignKey("ego.id"), primary_key=True),
    Column("map_id", ForeignKey("map.id"), primary_key=True),
)

agent_map = Table(
    "agent_map",
    Base.metadata,
    Column("agent_id", ForeignKey("agent.id"), primary_key=True),
    Column("map_id", ForeignKey("map.id"), primary_key=True),
)

frame_maneuver = Table(
    "frame_maneuver",
    Base.metadata,
    Column("frame_id", ForeignKey("frame.id"), primary_key=True),
    Column("maneuver", ForeignKey("maneuver.id"), primary_key=True),
)

agent_maneuver = Table(
    "agent_maneuver",
    Base.metadata,
    Column("agent_id", ForeignKey("agent.id"), primary_key=True),
    Column("maneuver", ForeignKey("maneuver.id"), primary_key=True),
)
