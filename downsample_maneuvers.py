import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
import numpy as np
import fpsample

from annotator.data.models import Frame, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType


def downsample_mans(session, max_nb_samples=50, d_threshold=30.0):
    stmt = select(Maneuver)
    maneuvers = session.scalars(stmt).all()

    for man in maneuvers:
        if man.is_ego:
            # print(f"Skipping {man.pos_maneuvers[0]!r}")
            continue

        if len(man.pos_maneuvers) > 0 and (
            man.pos_maneuvers[0].type == ManeuverType.U_TURN
            or man.pos_maneuvers[0].type == ManeuverType.MOVING_RIGHT_OF_AGENT
            or man.pos_maneuvers[0].type == ManeuverType.MOVING_LEFT_OF_AGENT
        ):
            # print(f"Skipping {man.pos_maneuvers[0]!r}")
            continue

        # threshold by visibility
        vis = man.get_visibilities()
        if vis[0] == VisibilityType.LOW:
            man.in_use = False

        # threshold by distance
        if np.linalg.norm(man.get_xys_in_ego(), axis=1).mean() > d_threshold:
            man.in_use = False

    for maneuver_type in ManeuverType:
        # skipt it for u-turns
        if maneuver_type == ManeuverType.U_TURN:
            continue

        stmt = (
            select(Maneuver)
            .where(Maneuver.in_use == True)
            .where(Maneuver.pos_maneuvers.any(PositiveManeuver.type == maneuver_type))
        )
        maneuvers = session.scalars(stmt).all()
        maneuvers = [
            man for man in maneuvers if not man.is_ego
        ]  # exclude ego maneuvers

        if len(maneuvers) < max_nb_samples:
            continue

        # fps sample based on xy location
        xys = np.stack([man.get_xys_in_ego() for man in maneuvers]).mean(axis=1)
        fps_samples_idx = fpsample.fps_sampling(xys, max_nb_samples)
        for i, man in enumerate(maneuvers):
            if i not in fps_samples_idx:
                man.in_use = False

    session.commit()


def main(args):
    engine = create_engine(f"sqlite:///{str(args.db_path.name)}", echo=False)
    session = Session(engine)

    downsample_mans(session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STSBench Annotator")
    parser.add_argument(
        "--db_path",
        type=Path,
        default="nuScenes.db",
    )
    args = parser.parse_args()

    main(args)
