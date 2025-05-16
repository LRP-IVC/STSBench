import argparse
from pathlib import Path
from typing import List

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
import numpy as np
from tqdm import tqdm

from annotator.data.models import Base, Scene, Frame, Agent, Ego, Track
from annotator.data.maneuvers import (
    ManeuverType,
    PositiveManeuver,
    NegativeManeuver,
    Maneuver,
)
from annotator.mining.ego import (
    is_accelerating,
    is_agent_overtaking_agent,
    is_agent_passing_agent,
    is_agent_following_agent,
    is_agent_leading_agent,
    is_agent_walking_alongside,
    is_agent_walking_opposite,
    is_decelerating,
    is_stopping,
    is_changing_lanes,
    is_truning_left,
    is_truning_right,
    is_u_turn,
    is_reversing,
    is_agent_waiting_cross,
    is_crossing,
    is_jaywalking,
    is_running,
    is_standing,
    is_walking,
    is_agent_stationary_behind_agent,
    is_agent_stationary_in_front_of_agent,
    is_agent_stationary_right_of_agent,
    is_agent_stationary_left_of_agent,
    is_agent_moving_right_of_agent,
    is_agent_moving_left_of_agent,
)
from annotator.mining.consts import NEGATIVE_MANEUVERS, PED_NEGATIVE_MANEUVERS

MINE_FUNC = {
    ManeuverType.ACCELERATE: is_accelerating,
    ManeuverType.DECELERATE: is_decelerating,
    ManeuverType.STOP: is_stopping,
    ManeuverType.LEFT_TURN: is_truning_left,
    ManeuverType.RIGHT_TURN: is_truning_right,
    ManeuverType.U_TURN: is_u_turn,
    ManeuverType.REVERSE: is_reversing,
    ManeuverType.LANE_CHANGE: is_changing_lanes,
    ManeuverType.OVERTAKE_AGENT: is_agent_overtaking_agent,
    ManeuverType.PASS_AGENT: is_agent_passing_agent,
    ManeuverType.OVERTAKE_EGO: is_agent_overtaking_agent,
    ManeuverType.FOLLOW_EGO: is_agent_following_agent,
    ManeuverType.FOLLOW_AGENT: is_agent_following_agent,
    ManeuverType.LEAD_AGENT: is_agent_leading_agent,
    ManeuverType.LEAD_EGO: is_agent_leading_agent,
    ManeuverType.PASS_EGO: is_agent_passing_agent,
    ManeuverType.WAIT_PED_CROSS: is_agent_waiting_cross,
    ManeuverType.CROSS: is_crossing,
    ManeuverType.JAYWALK: is_jaywalking,
    ManeuverType.RUN: is_running,
    ManeuverType.STAND: is_standing,
    ManeuverType.WALK: is_walking,
    ManeuverType.WALK_ALONGSIDE: is_agent_walking_alongside,
    ManeuverType.WALK_OPPOSITE: is_agent_walking_opposite,
    ManeuverType.STATIONARY_BEHIND_AGENT: is_agent_stationary_behind_agent,
    ManeuverType.STATIONARY_IN_FRONT_OF_AGENT: is_agent_stationary_in_front_of_agent,
    ManeuverType.STATIONARY_BEHIND_EGO: is_agent_stationary_behind_agent,
    ManeuverType.STATIONARY_IN_FRONT_OF_EGO: is_agent_stationary_in_front_of_agent,
    ManeuverType.STATIONARY_IN_FRONT_OF_EGO: is_agent_stationary_in_front_of_agent,
    ManeuverType.STATIONARY_RIGHT_OF_AGENT: is_agent_stationary_right_of_agent,
    ManeuverType.STATIONARY_LEFT_OF_AGENT: is_agent_stationary_left_of_agent,
    ManeuverType.STATIONARY_RIGHT_OF_EGO: is_agent_stationary_right_of_agent,
    ManeuverType.STATIONARY_LEFT_OF_EGO: is_agent_stationary_left_of_agent,
    ManeuverType.MOVING_RIGHT_OF_AGENT: is_agent_moving_right_of_agent,
    ManeuverType.MOVING_LEFT_OF_AGENT: is_agent_moving_left_of_agent,
    ManeuverType.MOVING_RIGHT_OF_EGO: is_agent_moving_right_of_agent,
    ManeuverType.MOVING_LEFT_OF_EGO: is_agent_moving_left_of_agent,
}


def mine_ego_maneuver(frames: List[Frame], maneuver_type: ManeuverType):
    egos = [f.ego for f in frames]
    for start_idx, end_idx in MINE_FUNC[maneuver_type](egos):
        if start_idx is None or end_idx is None:
            continue
        poss = [PositiveManeuver(type=maneuver_type)]
        negs = [NegativeManeuver(type=man) for man in NEGATIVE_MANEUVERS[maneuver_type]]
        yield Maneuver(
            frames=frames[start_idx:end_idx],
            manually_labeled=False,
            labeling_time=-1,
            in_use=True,
            pos_maneuvers=poss,
            prelabeled_pos_maneuvers=[p.type for p in poss],
            neg_maneuvers=negs,
            prelabeled_neg_maneuvers=[n.type for n in negs],
            is_ego=True,
            is_agent=False,
        )


def mine_ego_agent_maneuver(scene: Scene, session, maneuver_type: ManeuverType):
    stmt = select(Track).where(Track.scene_id == scene.id)
    other_tracks = session.scalars(stmt).all()

    for other_track in other_tracks:
        agents, egos = other_track.get_synchronized_ego()
        for start_idx, end_idx in MINE_FUNC[maneuver_type](egos, agents):
            if start_idx is None or end_idx is None:
                continue
            poss = [PositiveManeuver(type=maneuver_type)]
            negs = [
                NegativeManeuver(type=man) for man in NEGATIVE_MANEUVERS[maneuver_type]
            ]
            yield Maneuver(
                frames=[e.frame for e in egos[start_idx:end_idx]],
                other_agents=agents[start_idx:end_idx],
                manually_labeled=False,
                labeling_time=-1,
                in_use=True,
                pos_maneuvers=poss,
                prelabeled_pos_maneuvers=[p.type for p in poss],
                neg_maneuvers=negs,
                prelabeled_neg_maneuvers=[n.type for n in negs],
                is_ego=True,
                is_agent=False,
            )


def mine_agent_maneuver(track: Track, maneuver_type: ManeuverType):
    for start_idx, end_idx in MINE_FUNC[maneuver_type](track.agents):
        if start_idx is None or end_idx is None:
            continue
        poss = [PositiveManeuver(type=maneuver_type)]
        negs = [NegativeManeuver(type=man) for man in NEGATIVE_MANEUVERS[maneuver_type]]
        frames = [a.frame for a in track.agents[start_idx:end_idx]]
        yield Maneuver(
            frames=frames,
            instance_token=track.instance_token,
            manually_labeled=False,
            labeling_time=-1,
            in_use=True,
            pos_maneuvers=poss,
            prelabeled_pos_maneuvers=[p.type for p in poss],
            neg_maneuvers=negs,
            prelabeled_neg_maneuvers=[n.type for n in negs],
            is_ego=False,
            is_agent=True,
        )


def mine_agent_agent_maneuver(track: Track, session, maneuver_type: ManeuverType):
    # get tracks from the same scene
    stmt = select(Track).where(Track.scene_id == track.scene_id)
    other_tracks = session.scalars(stmt).all()

    for other_track in other_tracks:
        if track.id == other_track.id:
            continue

        cur_agents, other_agents = track.get_synchronized_agents(other_track)
        for start_idx, end_idx in MINE_FUNC[maneuver_type](cur_agents, other_agents):
            if start_idx is None or end_idx is None:
                continue
            poss = [PositiveManeuver(type=maneuver_type)]

            # override "standard" negative maneuvers with pedestrian-specific
            negative_maneuver = NEGATIVE_MANEUVERS
            if (
                cur_agents[0].is_human
                and other_agents[0].is_human
                and maneuver_type in PED_NEGATIVE_MANEUVERS.keys()
            ):
                negative_maneuver = PED_NEGATIVE_MANEUVERS

            negs = [
                NegativeManeuver(type=man) for man in negative_maneuver[maneuver_type]
            ]

            yield Maneuver(
                frames=[a.frame for a in cur_agents[start_idx:end_idx]],
                instance_token=track.instance_token,
                other_agents=other_agents[start_idx:end_idx],
                manually_labeled=False,
                labeling_time=-1,
                in_use=True,
                pos_maneuvers=poss,
                prelabeled_pos_maneuvers=[p.type for p in poss],
                neg_maneuvers=negs,
                prelabeled_neg_maneuvers=[n.type for n in negs],
                is_ego=False,
                is_agent=True,
            )


def mine_agent_ego_maneuver(track: Track, maneuver_type: ManeuverType):
    agents, egos = track.get_synchronized_ego()
    for start_idx, end_idx in MINE_FUNC[maneuver_type](agents, egos):
        if start_idx is None or end_idx is None:
            continue
        poss = [PositiveManeuver(type=maneuver_type)]
        negs = [NegativeManeuver(type=man) for man in NEGATIVE_MANEUVERS[maneuver_type]]
        yield Maneuver(
            frames=[a.frame for a in agents[start_idx:end_idx]],
            instance_token=track.instance_token,
            manually_labeled=False,
            labeling_time=-1,
            in_use=True,
            pos_maneuvers=poss,
            prelabeled_pos_maneuvers=[p.type for p in poss],
            neg_maneuvers=negs,
            prelabeled_neg_maneuvers=[n.type for n in negs],
            is_ego=False,
            is_agent=True,
        )


def mine_ego(args):
    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    Base.metadata.create_all(engine)
    session = Session(engine)

    stmt = select(Scene)
    scenes = session.scalars(stmt).all()

    for scene in tqdm(scenes):
        maneuver_types = [
            ManeuverType.ACCELERATE,
            ManeuverType.DECELERATE,
            ManeuverType.STOP,
            ManeuverType.LEFT_TURN,
            ManeuverType.RIGHT_TURN,
            ManeuverType.U_TURN,
            ManeuverType.REVERSE,
            ManeuverType.LANE_CHANGE,
        ]
        for maneuver_type in maneuver_types:
            for maneuver in mine_ego_maneuver(scene.frames, maneuver_type):
                session.add(maneuver)
            session.commit()

        maneuver_types = [
            ManeuverType.OVERTAKE_AGENT,
            ManeuverType.WAIT_PED_CROSS,
            ManeuverType.FOLLOW_AGENT,
            ManeuverType.LEAD_AGENT,
            ManeuverType.PASS_AGENT,
            ManeuverType.STATIONARY_BEHIND_AGENT,
            ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
            ManeuverType.STATIONARY_LEFT_OF_AGENT,
            ManeuverType.STATIONARY_RIGHT_OF_AGENT,
            ManeuverType.MOVING_RIGHT_OF_AGENT,
            ManeuverType.MOVING_LEFT_OF_AGENT,
        ]
        for maneuver_type in maneuver_types:
            for maneuver in mine_ego_agent_maneuver(scene, session, maneuver_type):
                session.add(maneuver)
            session.commit()


def mine_negs_agent(man: Maneuver):
    # for human objects, distinguishing between run, walk, and stand is straightforward.
    if not man.is_human:
        return man

    vel = [a.velocity for a in man.agents]
    med_vel = np.median(vel)

    if med_vel > 1.66:
        man.remove_negative(ManeuverType.RUN)
    elif 1.66 >= med_vel >= 0.5:
        man.remove_negative(ManeuverType.WALK)
    elif med_vel < 0.5:
        man.remove_negative(ManeuverType.STAND)

    return man


def mine_agent(args):
    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    Base.metadata.create_all(engine)
    session = Session(engine)

    stmt = select(Track)
    tracks = session.scalars(stmt).all()

    for track in tqdm(tracks):
        assert np.all(np.diff([a.frame.timestamp for a in track.agents]) > 0), (
            "Data is not timestamp-sorted"
        )
        maneuver_types = [
            ManeuverType.ACCELERATE,
            ManeuverType.STOP,
            ManeuverType.LEFT_TURN,
            ManeuverType.RIGHT_TURN,
            ManeuverType.U_TURN,
            ManeuverType.REVERSE,
            ManeuverType.LANE_CHANGE,
            ManeuverType.CROSS,
            ManeuverType.JAYWALK,
            ManeuverType.RUN,
            ManeuverType.STAND,
            ManeuverType.WALK,
        ]
        for maneuver_type in maneuver_types:
            for maneuver in mine_agent_maneuver(track, maneuver_type):
                session.add(maneuver)
                maneuver = mine_negs_agent(maneuver)
            session.commit()

        maneuver_types = [
            ManeuverType.OVERTAKE_AGENT,
            ManeuverType.WAIT_PED_CROSS,
            ManeuverType.FOLLOW_AGENT,
            ManeuverType.LEAD_AGENT,
            ManeuverType.WALK_ALONGSIDE,
            ManeuverType.WALK_OPPOSITE,
            ManeuverType.PASS_AGENT,
            ManeuverType.STATIONARY_BEHIND_AGENT,
            ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
            ManeuverType.STATIONARY_RIGHT_OF_AGENT,
            ManeuverType.STATIONARY_LEFT_OF_AGENT,
            ManeuverType.MOVING_RIGHT_OF_AGENT,
            ManeuverType.MOVING_LEFT_OF_AGENT,
        ]
        for maneuver_type in maneuver_types:
            for maneuver in mine_agent_agent_maneuver(track, session, maneuver_type):
                session.add(maneuver)
                maneuver = mine_negs_agent(maneuver)
            session.commit()

        maneuver_types = [
            ManeuverType.OVERTAKE_EGO,
            ManeuverType.PASS_EGO,
            ManeuverType.FOLLOW_EGO,
            ManeuverType.LEAD_EGO,
            ManeuverType.STATIONARY_BEHIND_EGO,
            ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
            ManeuverType.STATIONARY_RIGHT_OF_EGO,
            ManeuverType.STATIONARY_LEFT_OF_EGO,
            ManeuverType.MOVING_RIGHT_OF_EGO,
            ManeuverType.MOVING_LEFT_OF_EGO,
        ]
        for maneuver_type in maneuver_types:
            for maneuver in mine_agent_ego_maneuver(track, maneuver_type):
                session.add(maneuver)
            session.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Mining maneuvers", usage="%(prog)s [options]"
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    args = parser.parse_args()

    mine_ego(args)
    mine_agent(args)
