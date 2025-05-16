import enum
from typing import List, Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator, String
from sqlalchemy.ext.mutable import MutableList
import numpy as np

from . import (
    Base,
    frame_maneuver,
    agent_maneuver,
)


class ManeuverType(enum.IntEnum):
    ACCELERATE = 0
    DECELERATE = 1
    LANE_CHANGE = 2
    LEFT_TURN = 3
    RIGHT_TURN = 4
    U_TURN = 5
    REVERSE = 6
    STOP = 7
    #
    OVERTAKE_EGO = 20
    FOLLOW_EGO = 21
    LEAD_EGO = 22
    OVERTAKE_AGENT = 23
    WAIT_PED_CROSS = 24
    FOLLOW_AGENT = 25
    LEAD_AGENT = 26
    PASS_AGENT = 27
    PASS_EGO = 28
    STATIONARY_BEHIND_AGENT = 29
    STATIONARY_IN_FRONT_OF_AGENT = 30
    STATIONARY_BEHIND_EGO = 31
    STATIONARY_IN_FRONT_OF_EGO = 32
    STATIONARY_RIGHT_OF_AGENT = 33
    STATIONARY_LEFT_OF_AGENT = 34
    STATIONARY_RIGHT_OF_EGO = 35
    STATIONARY_LEFT_OF_EGO = 36
    MOVING_RIGHT_OF_AGENT = 37
    MOVING_LEFT_OF_AGENT = 38
    MOVING_RIGHT_OF_EGO = 39
    MOVING_LEFT_OF_EGO = 40
    #
    CROSS = 50
    JAYWALK = 51
    RUN = 52
    WALK = 53
    STAND = 54
    WALK_ALONGSIDE = 55
    WALK_OPPOSITE = 56


def ego_maneuvers() -> List[ManeuverType]:
    return [
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
    ]


class IntEnumList(TypeDecorator):
    """Represents a list of IntEnum values as a comma-separated string of integers."""

    impl = String

    def __init__(self, enum_class, **kwargs):
        super().__init__(**kwargs)
        self.enum_class = enum_class

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return ",".join([str(e.value) for e in value])

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return [self.enum_class(int(item)) for item in value.split(",")]


class NegativeManeuver(Base):
    __tablename__ = "negative_maneuver"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[ManeuverType]

    maneuver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("maneuver.id"))
    maneuver: Mapped[Optional["Maneuver"]] = relationship(
        back_populates="neg_maneuvers"
    )

    def __repr__(self) -> str:
        return f"NegativeManeuver(type={self.type.name!r})"

    def __eq__(self, other):
        return self.type == other.type

    def __lt__(self, other):
        return self.type.name < other.type.name


class PositiveManeuver(Base):
    __tablename__ = "positive_maneuver"
    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[ManeuverType]

    maneuver_id: Mapped[Optional[int]] = mapped_column(ForeignKey("maneuver.id"))
    maneuver: Mapped[Optional["Maneuver"]] = relationship(
        back_populates="pos_maneuvers"
    )

    def __repr__(self) -> str:
        return f"PositiveManeuver(type={self.type.name!r})"

    def __eq__(self, other):
        return self.type == other.type

    def __lt__(self, other):
        return self.type.name < other.type.name


class Maneuver(Base):
    __tablename__ = "maneuver"
    id: Mapped[int] = mapped_column(primary_key=True)
    manually_labeled: Mapped[bool]
    in_use: Mapped[bool]
    labeling_time: Mapped[Optional[int]]
    is_ego: Mapped[bool]
    is_agent: Mapped[bool]

    pos_maneuvers: Mapped[Optional[List[PositiveManeuver]]] = relationship(
        back_populates="maneuver"
    )
    neg_maneuvers: Mapped[Optional[List[NegativeManeuver]]] = relationship(
        back_populates="maneuver"
    )

    prelabeled_pos_maneuvers: Mapped[list[ManeuverType]] = mapped_column(
        MutableList.as_mutable(IntEnumList(ManeuverType))
    )
    prelabeled_neg_maneuvers: Mapped[list[ManeuverType]] = mapped_column(
        MutableList.as_mutable(IntEnumList(ManeuverType))
    )

    instance_token: Mapped[Optional[str]]
    frames: Mapped[List["Frame"]] = relationship(
        secondary=frame_maneuver,
        back_populates="maneuvers",
        order_by="Frame.timestamp",
    )

    other_agents: Mapped[List["Agent"]] = relationship(
        secondary=agent_maneuver,
        back_populates="maneuvers",
    )

    def __repr__(self) -> str:
        return f"Maneuver(pos_maneuvers={self.pos_maneuvers!r}, neg_maneuvers={self.neg_maneuvers!r})"

    @property
    def agents(self):
        agents = []
        for frame in self.frames:
            for agent in frame.agents:
                if agent.track.instance_token == self.instance_token:
                    agents.append(agent)
        return agents

    @property
    def egos(self):
        return [f.ego for f in self.frames]

    @property
    def actor(self) -> str:
        if self.instance_token is None:
            return "ego"
        else:
            return "agent"

    @property
    def is_human(self):
        if self.actor == "ego":
            return False
        return self.agents[0].is_human

    @property
    def is_other_human(self):
        if self.actor == "ego":
            return False
        return (
            self.agents[0].is_human
            and self.is_other_agent
            and self.other_agents[0].is_human
        )

    @property
    def is_other_agent(self):
        return self.other_agents is not None and len(self.other_agents) > 0

    def remove_negative(self, to_remove: ManeuverType):
        if self.neg_maneuvers is None:
            return
        to_remove_idx = -1
        for i, nm in enumerate(self.neg_maneuvers):
            if nm.type == to_remove:
                to_remove_idx = i
                break
        if to_remove_idx > 0:
            self.neg_maneuvers.pop(to_remove_idx)

    def get_xys_in_ego(self):
        assert self.is_agent, "It has to be agent's maneuver"
        xys = []
        for a, f in zip(self.agents, self.frames):
            xys.append(a.translate_rotate_xyz(-np.array(f.ego.xyz), f.ego.q.inverse))
        return np.array(xys)

    def get_other_xys_in_ego(self):
        assert self.is_other_agent, "It has to be two agent maneuver"
        xys = []
        for a, f in zip(self.other_agents, self.frames):
            xys.append(a.translate_rotate_xyz(-np.array(f.ego.xyz), f.ego.q.inverse))
        return np.array(xys)

    def get_visibilities(self):
        assert self.is_agent, "It has to be agent's maneuver"
        return np.array([a.visibility for a in self.agents])
