from typing import List

import numpy as np
from ..data.models import Ego, Agent, VisibilityType
from ..data import utils as adu


def merge_true_islands_center(data):
    """
    Merges consecutive True values in a list into a single True at the center
    of the island.  For even length islands, the right-center is chosen.

    Args:
        data: A list of boolean values.

    Returns:
        A new list with merged True islands.
    """

    n = len(data)
    result = [False] * n

    start_index = -1  # Start index of the current island
    island_length = 0  # Length of the current island

    for i in range(n):
        if data[i]:
            if island_length == 0:  # Start of a new island
                start_index = i
            island_length += 1
        else:  # Encountered a False value
            if island_length > 0:  # Process the previous island (if any)
                center_index = (
                    start_index + (island_length - 1) // 2
                )  # Integer division for center
                result[center_index] = True
            island_length = 0  # Reset for the next potential island

    # Handle any island that might be at the very end of the list
    if island_length > 0:
        center_index = start_index + (island_length - 1) // 2
        result[center_index] = True

    return result


def is_accelerating(
    objs: List[Ego | Agent], frames: int = 6, threshold_ms: float = 3.0
):
    if not objs[0].is_vehicle:
        return (None, None)

    vel = np.array([o.velocity for o in objs])

    if vel.shape[0] < frames:
        return (None, None)

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    acc_w = np.diff(vel_w, n=1, axis=1)
    mask = np.all(acc_w > 0.1, axis=1) & (vel_w[:, -1] - vel_w[:, 0] > threshold_ms)

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_decelerating(
    objs: List[Ego | Agent], frames: int = 6, threshold_ms: float = 3.0
):
    if not objs[0].is_vehicle:
        return (None, None)

    vel = np.array([o.velocity for o in objs])

    if vel.shape[0] < frames:
        return (None, None)

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    acc_w = np.diff(vel_w, n=1, axis=1)
    mask = (
        np.all(acc_w < 0.1, axis=1)  # decelerating between frames
        & (vel_w[:, 0] - vel_w[:, -1] > threshold_ms)  # speed diff threshold is reached
        & (vel_w[:, -1] > 1.5)  # is driving further (different from stop)
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_stopping(objs: List[Ego | Agent], frames: int = 6, threshold_ms: float = 3.0):
    if not objs[0].is_vehicle:
        return (None, None)

    vel = np.array([o.velocity for o in objs])

    if vel.shape[0] < frames:
        return (None, None)

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    acc_w = np.diff(vel_w, n=1, axis=1)
    mask = (
        np.all(acc_w < 0.1, axis=1)  # decelerating between frames
        & (vel_w[:, 0] - vel_w[:, -1] > threshold_ms)  # speed diff threshold is reached
        & (vel_w[:, -1] < 0.55)  # is stopped
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_agent_overtaking_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not (
        (objs[0].is_vehicle and other_objs[0].is_vehicle)
        or (objs[0].is_human and other_objs[0].is_human)
    ):
        return (None, None)

    is_human = objs[0].is_human

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))
    xy_in_other = np.array([
        o.translate_rotate_xyz(-np.array(oo.xyz), oo.q.inverse)
        for o, oo in zip(objs, other_objs)
    ])[:, 0:2]

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)
    delta_v = v_w - other_v_w

    mask = (
        np.all(delta_xy < (1.5 if is_human else 5.0), axis=1)  # close to other
        & np.all(delta_v > 0.0, axis=1)  # faster then other
        & (v_w.min(axis=1) > (0.5 if is_human else 2.0))  # moving
        & (other_v_w.min(axis=1) > (0.5 if is_human else 2.0))  # other is also moving
        & np.all(np.abs(yaw_w - other_yaw_w) < 20.0, axis=1)  # in the same direction
        & (xy_in_other_w[:, 0, 0] < 0)  # behind other at first
        & (xy_in_other_w[:, -1, 0] > 0)  # in front of other at the end
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_passing_agent(objs: List[Agent], other_objs: List[Agent], frames=6):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not (
        (objs[0].is_vehicle and other_objs[0].is_vehicle)
        or (objs[0].is_human and other_objs[0].is_human)
    ):
        return (None, None)

    is_human = objs[0].is_human

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))
    xy_in_other = np.array([
        o.translate_rotate_xyz(-np.array(oo.xyz), oo.q.inverse)
        for o, oo in zip(objs, other_objs)
    ])[:, 0:2]

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)
    mask = (
        (delta_xy.max(axis=1) < (2.0 if is_human else 10.0))  # close to other
        & (v_w.min(axis=1) > (0.5 if is_human else 2.0))  # moving
        & (other_v_w.max(axis=1) < (0.5 if is_human else 2.0))  # other is standing
        & np.all(np.abs(yaw_w - other_yaw_w) < 20.0, axis=1)  # in the same direction
        & (xy_in_other_w[:, 0, 0] < 0)  # behind other at first
        & (xy_in_other_w[:, -1, 0] > 0)  # in front of other at the end
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_following_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if isinstance(objs[0], Agent) and not objs[0].is_vehicle:
        return (None, None)

    if isinstance(other_objs[0], Agent) and not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)
    delta_v = v_w - other_v_w

    mask = (
        np.all(np.abs(xy_in_other_w[..., 1]) < 2.0, axis=1)  # lateral difference small
        & np.all(xy_in_other_w[..., 0] < 0.0, axis=1)  # behind other agent
        & np.all(delta_xy < 20.0, axis=1)  # close to other agent
        & np.all(np.abs(delta_v) < 3.0, axis=1)  # same speed
        & (v_w.min(axis=1) > 2.0)  # moving
        & (other_v_w.min(axis=1) > 2.0)  # other agent is also moving
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_leading_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if isinstance(objs[0], Agent) and not objs[0].is_vehicle:
        return (None, None)

    if isinstance(other_objs[0], Agent) and not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)
    delta_v = v_w - other_v_w

    mask = (
        np.all(np.abs(xy_in_other_w[..., 1]) < 2.0, axis=1)  # lateral difference small
        & np.all(xy_in_other_w[..., 0] > 0.0, axis=1)  # in front of other agent
        & np.all(delta_xy < 20.0, axis=1)  # close to other agent
        & np.all(np.abs(delta_v) < 3.0, axis=1)  # same speed
        & (v_w.min(axis=1) > 2.0)  # moving
        & (other_v_w.min(axis=1) > 2.0)  # other agent is also moving
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_changing_lanes(objs: List[Ego | Agent], frames: int = 6):
    if not objs[0].is_vehicle:
        return (None, None)

    lanes = [o.get_lane_id for o in objs]
    lanes = np.array(lanes, dtype=float)

    if lanes.size < frames:
        return (None, None)

    lanes[lanes == -1] = np.nan

    lane_w = np.lib.stride_tricks.sliding_window_view(lanes, frames, axis=0)
    lane_delta = lane_w[:, :-1] - lane_w[:, 1:]
    lane_delta = np.abs(np.nan_to_num(lane_delta))

    mask = np.any(lane_delta > 0, axis=1)

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_truning_left(
    objs: List[Ego | Agent], frames: int = 6, threshold_rad: float = 0.8
):
    if not objs[0].is_vehicle:
        return (None, None)
    yaw = np.array([o.yaw for o in objs])

    if yaw.size < frames:
        return (None, None)

    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)
    yaw_w_diff = np.arctan2(
        np.sin(yaw_w[:, -1] - yaw_w[:, 0]),
        np.cos(yaw_w[:, -1] - yaw_w[:, 0]),
    )

    mask = yaw_w_diff > threshold_rad

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_truning_right(
    objs: List[Ego | Agent], frames: int = 6, threshold_rad: float = 0.8
):
    if not objs[0].is_vehicle:
        return (None, None)

    yaw = np.array([o.yaw for o in objs])

    if yaw.size < frames:
        return (None, None)

    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)
    yaw_w_diff = np.arctan2(
        np.sin(yaw_w[:, -1] - yaw_w[:, 0]),
        np.cos(yaw_w[:, -1] - yaw_w[:, 0]),
    )

    mask = yaw_w_diff < -threshold_rad

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_u_turn(objs: List[Ego | Agent], frames: int = 6, threshold_rad: float = 1.7):
    if not objs[0].is_vehicle:
        return (None, None)

    yaw = np.array([o.yaw for o in objs])

    if yaw.size < frames:
        return (None, None)

    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)
    yaw_w_diff = np.arctan2(
        np.sin(yaw_w[:, -1] - yaw_w[:, 0]),
        np.cos(yaw_w[:, -1] - yaw_w[:, 0]),
    )

    mask = np.abs(yaw_w_diff) > threshold_rad

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_reversing(objs: List[Ego | Agent], frames: int = 6):
    if not objs[0].is_vehicle:
        return (None, None)

    vel = np.array([o.velocity for o in objs])
    yaw = np.array([o.yaw for o in objs])

    if yaw.size < frames:
        return (None, None)

    # estimate yaw
    xyz = np.array([o.xyz for o in objs])
    delta = xyz[1:, :2] - xyz[:-1, :2]
    est_yaw = np.arctan2(delta[:, 1], delta[:, 0])
    # extrapolate last value
    coefficients_quad = np.polyfit(np.arange(len(est_yaw)), est_yaw, 2)
    polynomial_quad = np.poly1d(coefficients_quad)
    extrapolated_value_quad = polynomial_quad(len(est_yaw))
    est_yaw = np.append(est_yaw, extrapolated_value_quad)

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)
    est_yaw_w = np.lib.stride_tricks.sliding_window_view(est_yaw, frames, axis=0)

    yaw_w_delta = np.arctan2(
        np.sin(yaw_w - est_yaw_w),
        np.cos(yaw_w - est_yaw_w),
    )

    mask = (
        (
            np.count_nonzero(np.abs(yaw_w_delta) > 1.0, axis=1) > frames // 2
        )  # yaw deviation in >50% frames
        & (vel_w.mean(axis=1) > 1.5)  # has to move
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_agent_waiting_cross(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if isinstance(objs[0], Agent) and not objs[0].is_vehicle:
        return (None, None)

    if isinstance(other_objs[0], Agent) and not other_objs[0].is_human:
        return (None, None)

    xy = np.array([o.xy for o in objs])
    vel = np.array([o.velocity for o in objs])

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)

    other_xy = np.array([o.xy for o in other_objs])
    other_ped_crossing = [o.ped_crossing_id for o in other_objs]

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_ped_crossing_w = np.lib.stride_tricks.sliding_window_view(
        other_ped_crossing, frames, axis=0
    )

    delta_xy_w = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        (vel_w.mean(axis=1) < 0.55)  # vehicle almost stopped
        & (delta_xy_w.max(axis=-1) < 10.0)  # close to each other
        & np.any(other_ped_crossing_w > 0, axis=1)  # pedestrian on a crosswalk
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_crossing(objs: List[Agent], frames: int = 6):
    if len(objs) < frames or not objs[0].is_human:
        return (None, None)

    vel = np.array([o.velocity for o in objs])
    ped_crossing = np.array([o.ped_crossing_id for o in objs])
    dist_from_ego = np.array([o.dist_from_ego for o in objs])

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    ped_crossing_w = np.lib.stride_tricks.sliding_window_view(
        ped_crossing, frames, axis=0
    )
    dist_from_ego_w = np.lib.stride_tricks.sliding_window_view(
        dist_from_ego, frames, axis=0
    )

    mask = (
        np.all(ped_crossing_w > 0, axis=1)  # on pedestrian crossing
        & (vel_w.min(axis=1) > 0.5)  # walking instead of standing
        & (dist_from_ego_w.max(axis=1) < 40)  # not too far away from ego
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_jaywalking(objs: List[Agent], frames: int = 6):
    if len(objs) < frames or not objs[0].is_human:
        return (None, None)

    vel = np.array([o.velocity for o in objs])
    ped_crossing = np.array([o.ped_crossing_id for o in objs])
    drivable_area = np.array([o.drivable_area_id for o in objs])
    dist_from_ego = np.array([o.dist_from_ego for o in objs])

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    ped_crossing_w = np.lib.stride_tricks.sliding_window_view(
        ped_crossing, frames, axis=0
    )
    drivable_area_w = np.lib.stride_tricks.sliding_window_view(
        drivable_area, frames, axis=0
    )
    dist_from_ego_w = np.lib.stride_tricks.sliding_window_view(
        dist_from_ego, frames, axis=0
    )

    mask = (
        np.all(~(ped_crossing_w > 0), axis=1)  # not on pedestrian crossing
        & np.all(drivable_area_w > 0, axis=1)  # on pedestrian crossing
        & (vel_w.min(axis=1) > 0.5)  # walking instead of standing
        & (dist_from_ego_w.max(axis=1) < 40)  # not too far away from ego
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_running(objs: List[Agent], frames: int = 6):
    if len(objs) < frames or not objs[0].is_human:
        return (None, None)

    vel = np.array([o.velocity for o in objs])
    dist_from_ego = np.array([o.dist_from_ego for o in objs])

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    dist_from_ego_w = np.lib.stride_tricks.sliding_window_view(
        dist_from_ego, frames, axis=0
    )

    mask = (
        (vel_w.min(axis=1) > 2.5)  # walking instead of standing
        & (dist_from_ego_w.max(axis=1) < 40)  # not too far away from ego
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_standing(objs: List[Agent], frames: int = 6):
    if len(objs) < frames or not objs[0].is_human:
        return (None, None)

    vel = np.array([o.velocity for o in objs])

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)

    mask = (
        vel_w.max(axis=1) < 0.1  # standing
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_walking(objs: List[Agent], frames: int = 6):
    if len(objs) < frames or not objs[0].is_human:
        return (None, None)

    vel = np.array([o.velocity for o in objs])
    walkway = np.array([o.walkway_id for o in objs])

    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    walkway_w = np.lib.stride_tricks.sliding_window_view(walkway, frames, axis=0)

    mask = (
        np.all(walkway_w > 0, axis=1)  # on a walkway
        & (vel_w.min(axis=1) > 1.1)  # walking instead of standing
        & (vel_w.max(axis=1) < 1.6)  # but not too fast
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_agent_walking_alongside(objs: List[Agent], other_objs: List[Agent], frames=6):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if isinstance(objs[0], Agent) and not objs[0].is_human:
        return (None, None)

    if isinstance(other_objs[0], Agent) and not other_objs[0].is_human:
        return (None, None)

    xy = np.array([o.xy for o in objs])
    vel = np.array([o.velocity for o in objs])
    yaw = np.rad2deg(adu.get_yaw(objs))

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy = np.array([o.xy for o in other_objs])
    other_vel = np.array([o.velocity for o in other_objs])
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_vel_w = np.lib.stride_tricks.sliding_window_view(other_vel, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy_w = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        (delta_xy_w.max(axis=-1) < 1.0)  # close to each other
        & (vel_w.min(axis=1) > 1.1)  # walking
        & (other_vel_w.min(axis=1) > 1.1)  # other is also walking
        & np.all(np.abs(yaw_w - other_yaw_w) < 10.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_agent_walking_opposite(objs: List[Agent], other_objs: List[Agent], frames=6):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if isinstance(objs[0], Agent) and not objs[0].is_human:
        return (None, None)

    if isinstance(other_objs[0], Agent) and not other_objs[0].is_human:
        return (None, None)

    xy = np.array([o.xy for o in objs])
    vel = np.array([o.velocity for o in objs])
    yaw = np.rad2deg(adu.get_yaw(objs))

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    vel_w = np.lib.stride_tricks.sliding_window_view(vel, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy = np.array([o.xy for o in other_objs])
    other_vel = np.array([o.velocity for o in other_objs])
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_vel_w = np.lib.stride_tricks.sliding_window_view(other_vel, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy_w = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        (delta_xy_w.max(axis=-1) < 5.0)  # close to each other
        & (vel_w.min(axis=1) > 1.1)  # walking
        & (other_vel_w.min(axis=1) > 1.1)  # other is also walking
        & np.all(
            np.abs(yaw_w - other_yaw_w) > 150.0, axis=1
        )  # in the opposite direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield (start_idx, end_idx)


def is_agent_stationary_behind_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(np.abs(xy_in_other_w[..., 1]) < 2.0, axis=1)  # lateral difference small
        & np.all(xy_in_other_w[..., 0] < -1.0, axis=1)  # behind other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.max(axis=1) < 0.3)  # stationary
        & (other_v_w.max(axis=1) < 0.3)  # other agent is also stationary
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_stationary_in_front_of_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(np.abs(xy_in_other_w[..., 1]) < 2.0, axis=1)  # lateral difference small
        & np.all(xy_in_other_w[..., 0] > 1.0, axis=1)  # in front of other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.max(axis=1) < 0.3)  # stationary
        & (other_v_w.max(axis=1) < 0.3)  # other agent is also stationary
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_stationary_right_of_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(
            np.abs(xy_in_other_w[..., 0]) < 1.0, axis=1
        )  # longitudinal difference small
        & np.all(xy_in_other_w[..., 1] < -1.0, axis=1)  # right of other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.max(axis=1) < 0.3)  # stationary
        & (other_v_w.max(axis=1) < 0.3)  # other agent is also stationary
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_stationary_left_of_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(
            np.abs(xy_in_other_w[..., 0]) < 1.0, axis=1
        )  # longitudinal difference small
        & np.all(xy_in_other_w[..., 1] > 1.0, axis=1)  # right of other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.max(axis=1) < 0.3)  # stationary
        & (other_v_w.max(axis=1) < 0.3)  # other agent is also stationary
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_moving_right_of_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(
            np.abs(xy_in_other_w[..., 0]) < 1.0, axis=1
        )  # longitudinal difference small
        & np.all(xy_in_other_w[..., 1] < -1.0, axis=1)  # right of other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.min(axis=1) > 1.5)  # moving
        & (other_v_w.min(axis=1) > 1.5)  # other agent is also moving
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx


def is_agent_moving_left_of_agent(
    objs: List[Agent] | List[Ego], other_objs: List[Agent], frames=6
):
    if (len(objs) < frames) or (len(other_objs) < frames):
        return (None, None)

    if not objs[0].is_vehicle:
        return (None, None)

    if not other_objs[0].is_vehicle:
        return (None, None)

    xy = adu.get_xy(objs)
    v = adu.get_vel(objs)
    yaw = np.rad2deg(adu.get_yaw(objs))

    other_xy = adu.get_xy(other_objs)
    other_v = adu.get_vel(other_objs)
    other_yaw = np.rad2deg(adu.get_yaw(other_objs))

    # xy in ego
    xy_in_other = [
        o.translate_rotate_xyz(-np.array(e.xyz), e.q.inverse)
        for o, e in zip(objs, other_objs)
    ]

    xy_w = np.lib.stride_tricks.sliding_window_view(xy, frames, axis=0).transpose(
        0, 2, 1
    )
    xy_in_other_w = np.lib.stride_tricks.sliding_window_view(
        xy_in_other, frames, axis=0
    ).transpose(0, 2, 1)
    v_w = np.lib.stride_tricks.sliding_window_view(v, frames, axis=0)
    yaw_w = np.lib.stride_tricks.sliding_window_view(yaw, frames, axis=0)

    other_xy_w = np.lib.stride_tricks.sliding_window_view(
        other_xy, frames, axis=0
    ).transpose(0, 2, 1)
    other_v_w = np.lib.stride_tricks.sliding_window_view(other_v, frames, axis=0)
    other_yaw_w = np.lib.stride_tricks.sliding_window_view(other_yaw, frames, axis=0)

    delta_xy = np.linalg.norm(xy_w - other_xy_w, axis=-1)

    mask = (
        np.all(
            np.abs(xy_in_other_w[..., 0]) < 1.0, axis=1
        )  # longitudinal difference small
        & np.all(xy_in_other_w[..., 1] > 1.0, axis=1)  # left of other agent
        & np.all(delta_xy < 5.0, axis=1)  # close to other agent
        & (v_w.min(axis=1) > 1.5)  # moving
        & (other_v_w.min(axis=1) > 1.5)  # other agent is also moving
        & np.all(np.abs(yaw_w - other_yaw_w) < 15.0, axis=1)  # in the same direction
    )

    mask = merge_true_islands_center(mask)
    for start_idx in np.where(mask)[0]:
        end_idx = start_idx + frames
        yield start_idx, end_idx
