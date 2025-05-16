from typing import List, Dict

import numpy as np
from pyquaternion import Quaternion

from .models import Ego, Agent, Sensor, SensorType, Scene, Frame


def quaternion_yaw(q: Quaternion) -> float:
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


def get_xyz(objs: List[Ego]):
    return np.array([[obj.x, obj.y, obj.z] for obj in objs])


def get_xy(objs: List[Ego] | List[Agent]):
    return np.array([[obj.x, obj.y] for obj in objs])


def get_wlh(objs: List[Agent]):
    return np.array([[obj.w, obj.l, obj.h] for obj in objs])


def get_q_xyzw(objs: List[Ego]):
    return np.array([[obj.qx, obj.qy, obj.qz, obj.qw] for obj in objs])


def get_yaw(objs: List[Agent] | List[Ego]):
    return np.array([quaternion_yaw(Quaternion(*o.qwxyz)) for o in objs])


def get_timestamp(objs: List[Frame]):
    return np.array([obj.timestamp for obj in objs])


def get_vel(objs: List[Agent] | List[Ego]):
    if isinstance(objs[0], Agent):
        vel = np.array([[obj.vx, obj.vy] for obj in objs])
        return np.linalg.norm(vel, axis=1)
    elif isinstance(objs[0], Ego):
        return np.array([e.velocity for e in objs])
    else:
        raise ValueError("Object not supported")


def compute_vel(objs: List[Ego]):
    xy = get_xy(objs)
    ts = get_timestamp([e.frame for e in objs])
    vel = np.linalg.norm(xy[1:] - xy[:-1], axis=1, ord=2) / ((ts[1:] - ts[:-1]) * 1e-6)
    # extrapolate last value
    coefficients_quad = np.polyfit(np.arange(len(vel)), vel, 2)
    polynomial_quad = np.poly1d(coefficients_quad)
    extrapolated_value_quad = polynomial_quad(len(vel))
    vel = np.append(vel, extrapolated_value_quad)
    return vel


def get_sensors(objs: List[Agent]):
    sensors = {}
    for obj in objs:
        for sensor in obj.sensors:
            if sensor.type not in sensors:
                sensors[sensor.type] = []
            sensors[sensor.type].append(sensor)
    return sensors
