import argparse
import json
from pathlib import Path
from typing import Tuple
import random
import string

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from annotator.data.maneuvers import Maneuver, ego_maneuvers
from annotator.data.models import Frame, SensorType, VisibilityType
from annotator.data.maneuvers import Maneuver, PositiveManeuver, ManeuverType
from vqa_extractor.base import VQAExtractor


def short_maneuver_description(
    man_type,
    is_ego,
    is_agent,
    ego_desc="you",
    agent_desc="agent",
    other_agent_desc="agent",
) -> str:
    if man_type == ManeuverType.ACCELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} are accelerating"
        elif is_agent:
            return f"{agent_desc.capitalize()} is accelerating"
        else:
            assert False
    elif man_type == ManeuverType.DECELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} are decelerating"
        elif is_agent:
            return f"{agent_desc.capitalize()} is decelerating"
        else:
            assert False
    elif man_type == ManeuverType.LANE_CHANGE:
        if is_ego:
            return f"{ego_desc.capitalize()} are changing lanes"
        elif is_agent:
            return f"{agent_desc.capitalize()} are changing lanes"
        else:
            assert False
    elif man_type == ManeuverType.LEFT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are turning left"
        elif is_agent:
            return f"{agent_desc.capitalize()} is turning left"
        else:
            assert False
    elif man_type == ManeuverType.RIGHT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are turning right"
        elif is_agent:
            return f"{agent_desc.capitalize()} is turning right"
        else:
            assert False
    elif man_type == ManeuverType.U_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are performing u-turn"
        elif is_agent:
            return f"{agent_desc.capitalize()} is performing u-turn"
        else:
            assert False
    elif man_type == ManeuverType.REVERSE:
        if is_ego:
            return f"{ego_desc.capitalize()} are reversing"
        elif is_agent:
            return f"{agent_desc.capitalize()} is reversing"
        else:
            assert False
    elif man_type == ManeuverType.STOP:
        if is_ego:
            return f"{ego_desc.capitalize()} are stopping"
        elif is_agent:
            return f"{agent_desc.capitalize()} is stopping"
        else:
            assert False
    elif man_type == ManeuverType.OVERTAKE_EGO:
        return f"{agent_desc} is overtaking {ego_desc}".capitalize()
    elif man_type == ManeuverType.FOLLOW_EGO:
        return f"{agent_desc} is following {ego_desc}".capitalize()
    elif man_type == ManeuverType.LEAD_EGO:
        return f"{agent_desc} is leading {ego_desc}".capitalize()
    elif man_type == ManeuverType.PASS_EGO:
        return f"{agent_desc} is passes stationary {ego_desc}".capitalize()
    elif man_type == ManeuverType.OVERTAKE_AGENT:
        if is_ego:
            return f"{ego_desc} are overtaking {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is overtaking {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WAIT_PED_CROSS:
        if is_ego:
            return f"{ego_desc} are waiting for pedestrian to cross".capitalize()
        elif is_agent:
            return f"{agent_desc} is waiting for pedestrian to cross".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.FOLLOW_AGENT:
        if is_ego:
            return f"{ego_desc} are following {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is following {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.LEAD_AGENT:
        if is_ego:
            return f"{ego_desc} are leading {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is leading {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.PASS_AGENT:
        if is_ego:
            return f"{ego_desc} are passing stationary {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is passing stationary {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.CROSS:
        if is_agent:
            return f"{agent_desc} is crossing street".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.JAYWALK:
        if is_agent:
            return f"{agent_desc} is jaywalking".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.RUN:
        if is_agent:
            return f"{agent_desc} is running".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WALK:
        if is_agent:
            return f"{agent_desc} is walking".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STAND:
        if is_agent:
            return f"{agent_desc} is stationary".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WALK_ALONGSIDE:
        if is_agent:
            return f"{agent_desc} is walking alongside {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WALK_OPPOSITE:
        if is_agent:
            return f"{agent_desc} is walking in opposite direction of {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_BEHIND_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary behind {other_agent_desc}".capitalize()
        elif is_ego:
            return f"{ego_desc} are stationary behind {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary in front of {other_agent_desc}".capitalize()
        elif is_ego:
            return (
                f"{ego_desc} are stationary in front of {other_agent_desc}".capitalize()
            )
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_BEHIND_EGO:
        if is_agent:
            return f"{agent_desc} is stationary behind {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_EGO:
        if is_agent:
            return f"{agent_desc} are stationary in front of {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_RIGHT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary to the right of {other_agent_desc}".capitalize()
        elif is_ego:
            return f"{ego_desc} are stationary to the right of {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary to the left of {other_agent_desc}".capitalize()
        elif is_ego:
            return f"{ego_desc} are stationary to the left of {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_RIGHT_OF_EGO:
        if is_agent:
            return f"{agent_desc} is stationary to the right of {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_EGO:
        if is_agent:
            return f"{agent_desc} is stationary to the left of {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.MOVING_RIGHT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is moving to the right of {other_agent_desc}".capitalize()
        elif is_ego:
            return (
                f"{ego_desc} are moving to the right of {other_agent_desc}".capitalize()
            )
        else:
            assert False
    elif man_type == ManeuverType.MOVING_LEFT_OF_AGENT:
        if is_agent:
            return (
                f"{agent_desc} is moving to the left of {other_agent_desc}".capitalize()
            )
        elif is_ego:
            return (
                f"{ego_desc} are moving to the left of {other_agent_desc}".capitalize()
            )
        else:
            assert False
    elif man_type == ManeuverType.MOVING_RIGHT_OF_EGO:
        if is_agent:
            return f"{agent_desc} is moving to the right of {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.MOVING_LEFT_OF_EGO:
        if is_agent:
            return f"{agent_desc} is moving to the left of {ego_desc}".capitalize()
        else:
            assert False
    assert False, "Missing maneuver description"


def long_maneuver_description(
    man_type,
    is_ego,
    is_agent,
    ego_desc="you",
    agent_desc="object 1",
    other_agent_desc="object 2",
) -> str:
    if man_type == ManeuverType.ACCELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} are increasing your speed, either gradually or abruptly, to adapt to traffic conditions, maintain flow, or comply with traffic rules and signals."
        elif is_agent:
            return f"{agent_desc.capitalize()} is increasing its speed, either gradually or abruptly, to adapt to traffic conditions, maintain flow, or comply with traffic rules and signals."
        else:
            assert False
    elif man_type == ManeuverType.DECELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} are reducing your speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules, without coming to a complete stop."
        elif is_agent:
            return f"{agent_desc.capitalize()} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules, without coming to a complete stop."
        else:
            assert False
    elif man_type == ManeuverType.LANE_CHANGE:
        if is_ego:
            return f"{ego_desc.capitalize()} are transitioning from your current lane to an adjacent lane."
        elif is_agent:
            return f"{agent_desc.capitalize()} is transitioning from its current lane to an adjacent lane."
        else:
            assert False
    elif man_type == ManeuverType.LEFT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are executing a left turn at an intersection or junction."
        elif is_agent:
            return f"{agent_desc.capitalize()} is executing a left turn at an intersection or junction."
        else:
            assert False
    elif man_type == ManeuverType.RIGHT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are executing a right turn at an intersection or junction."
        elif is_agent:
            return f"{agent_desc.capitalize()} is executing a right turn at an intersection or junction."
        else:
            assert False
    elif man_type == ManeuverType.U_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} are performing a 180-degree turn at an intersection or junction, reversing its direction of travel."
        elif is_agent:
            return f"{agent_desc.capitalize()} is performing a 180-degree turn at an intersection or junction, reversing its direction of travel."
        else:
            assert False
    elif man_type == ManeuverType.REVERSE:
        if is_ego:
            return f"{ego_desc.capitalize()} are moving in reverse, either to park, navigate a tight space, or adjust your position."
        elif is_agent:
            return f"{agent_desc.capitalize()} is moving in reverse, either to park, navigate a tight space, or adjust its position."
        else:
            assert False
    elif man_type == ManeuverType.STOP:
        if is_ego:
            return f"{ego_desc.capitalize()} are reducing your speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules and comes to a complete stop."
        elif is_agent:
            return f"{agent_desc.capitalize()} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules and comes to a complete stop."
        else:
            assert False
    elif man_type == ManeuverType.OVERTAKE_EGO:
        return f"{agent_desc.capitalize()} in the adjacent lane moves ahead of {ego_desc} while both are in motion."
    elif man_type == ManeuverType.FOLLOW_EGO:
        return f"{agent_desc.capitalize()} is driving behind {ego_desc} at a similar speed while maintaining a consistent distance."
    elif man_type == ManeuverType.LEAD_EGO:
        return f"{agent_desc.capitalize()} travels ahead of {ego_desc} at a similar speed while maintaining a consistent distance."
    elif man_type == ManeuverType.PASS_EGO:
        return f"{agent_desc.capitalize()} in the adjacent lane overtakes stopped {ego_desc}."
    elif man_type == ManeuverType.OVERTAKE_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} are the adjacent lane and move ahead of {other_agent_desc} while both are in motion."
        elif is_agent:
            return f"{agent_desc.capitalize()} in the adjacent lane and moves ahead of {other_agent_desc} while both are in motion."
        else:
            assert False
    elif man_type == ManeuverType.WAIT_PED_CROSS:
        if is_ego:
            return f"{ego_desc.capitalize()} come to a stop or remain stationary, yielding the right-of-way to {other_agent_desc} who is crossing or preparing to cross the road, while maintaining awareness of the {other_agent_desc}'s movement and ensuring a safe distance until the crossing is complete."
        elif is_agent:
            return f"{agent_desc.capitalize()} comes to a stop or remains stationary, yielding the right-of-way to a {other_agent_desc} who is crossing or preparing to cross the road, while maintaining awareness of the {other_agent_desc}'s movement and ensuring a safe distance until the crossing is complete."
        else:
            assert False
    elif man_type == ManeuverType.FOLLOW_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} are driving behind {other_agent_desc} at a similar speed while maintaining a consistent distance."
        elif is_agent:
            return f"{agent_desc.capitalize()} is driving behind {other_agent_desc} at a similar speed while maintaining a consistent distance."
        else:
            assert False
    elif man_type == ManeuverType.LEAD_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} travel ahead of {other_agent_desc} at a similar speed while maintaining a consistent distance."
        elif is_agent:
            return f"{agent_desc.capitalize()} travels ahead of {other_agent_desc} at a similar speed while maintaining a consistent distance."
        else:
            assert False
    elif man_type == ManeuverType.PASS_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} in the adjacent lane overtakes the stopped {other_agent_desc}."
        elif is_agent:
            return f"{agent_desc.capitalize()} in the adjacent lane overtakes the stopped {other_agent_desc}."
        else:
            assert False
    elif man_type == ManeuverType.CROSS:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) moves from one side of the road to the other, at a designated crossing point or intersection."
        else:
            assert False
    elif man_type == ManeuverType.JAYWALK:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) crosses the street outside of designated crossing areas or against traffic signals, often requiring heightened awareness of vehicle movements, quick decision-making to avoid conflicts, and potentially creating unpredictable interactions with other agents in the traffic environment."
        else:
            assert False
    elif man_type == ManeuverType.RUN:
        if is_agent:
            return (
                f"{agent_desc.capitalize()} (pedestrian) is running and moves rapidly."
            )
        else:
            assert False
    elif man_type == ManeuverType.WALK:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) moves at a steady, moderate pace, typically following designated paths or crosswalks."
        else:
            assert False
    elif man_type == ManeuverType.STAND:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) remains stationary in the traffic environment, either waiting at a crossing, observing surroundings, or pausing for other reasons.".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WALK_ALONGSIDE:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) and {other_agent_desc} (pedestrian) walk side by side at a steady, moderate pace."
        else:
            assert False
    elif man_type == ManeuverType.WALK_OPPOSITE:
        if is_agent:
            return f"{agent_desc.capitalize()} (pedestrian) and {other_agent_desc} (pedestrian) walk toward each other at a moderate pace, cross paths, and proceed."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_BEHIND_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary behind {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        elif is_ego:
            return f"{agent_desc.capitalize()} are fully stopped and remain stationary behind {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary in front of {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        elif is_ego:
            return f"{ego_desc.capitalize()} are fully stopped and remain stationary in front of {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_BEHIND_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary behind {ego_desc} (which is also stopped), such as when waiting at a traffic light, in a parking lot, or in any other queuing scenario."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary ahead of {ego_desc} (which is also stopped), such as when waiting at a traffic light, in a parking lot, or in any other queuing scenario."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_RIGHT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary to the right of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        elif is_ego:
            return f"{ego_desc.capitalize()} are fully stopped and remain stationary to the right of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary to the left of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        elif is_ego:
            return f"{ego_desc.capitalize()} are fully stopped and remain stationary to the left of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_RIGHT_OF_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary to the right of {ego_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary to the left of {ego_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        else:
            assert False
    elif man_type == ManeuverType.MOVING_RIGHT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is traveling in parallel to the right of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        elif is_ego:
            return f"{ego_desc.capitalize()} are traveling in parallel to the right of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        else:
            assert False
    elif man_type == ManeuverType.MOVING_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is traveling in parallel to the left of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        elif is_ego:
            return f"{ego_desc.capitalize()} are traveling in parallel to the left of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        else:
            assert False
    elif man_type == ManeuverType.MOVING_RIGHT_OF_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is traveling in parallel to the right of {ego_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        else:
            assert False
    elif man_type == ManeuverType.MOVING_LEFT_OF_EGO:
        if is_agent:
            return f"{agent_desc.capitalize()} is traveling in parallel to the left of {ego_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        else:
            assert False
    assert False, "Missing maneuver description"


class SennaVQAExtractor(VQAExtractor):
    def __init__(self, number_of_negatives: int = 4):
        super().__init__(
            number_of_negatives=number_of_negatives,
            ego_desc="you",
            agent_desc="object 1",
            other_agent_desc="object 2",
            long_desc_generator=long_maneuver_description,
        )

        self.prompt_template.preamble = """A chat between a curious human and an artificial intelligence assistant. The assistent is specilized in the analysis and identification of temporal actions and maneuvers performed by various agents, as well as you, across diverse driving scenarios. Agents refer to all participants in the traffic environment, including but not limited to: cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles. You are the primary vehicle from whose perspective the scenario is being evaluated. You are equipped with a sophisticated suite of sensors (e.g., cameras, LiDAR, radar) to capture the surrounding traffic environment. Temporal actions and maneuvers include any time-based behaviors or movements, such as lane changes, accelerations, decelerations, turns, stops, or interactions between agents and between you and agents."""
        self.prompt_template.maneuver_description = "The following are driving maneuvers and actions along with their respective descriptions: {man_desc}The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <FRONT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\n<BACK VIEW>:\n<image>"
        self.prompt_template.postamble = (
            "For example, a correct answer format is like 'A'. ASSISTANT:"
        )
        self.prompt_template.ego_ref = "You are driving, which of the following options best describes your driving maneuver?"
        self.prompt_template.ego_ref_other_agent_ref = "I will now provide you with the position and velocity information of the dynamic objects:\nObject 2: {cls_2}, {long_desc_2}, {lat_desc_2}, speed of {speed_ms_2} m/s.\nPlease predict which of the following options best describes your driving behavior with respect to Object 2."
        self.prompt_template.agent_ref = "I will now provide you with the position and velocity information of the dynamic objects:\nObject 1: {cls}, {long_desc}, {lat_desc}, speed of {speed_ms} m/s.\nPlease predict which of the following options best describes Object 1 driving behavior."
        self.prompt_template.agent_ref_other_agent_ref = "I will now provide you with the position and velocity information of the dynamic objects:\nObject 1: {cls_1}, {long_desc_1}, {lat_desc_1}, speed of {speed_ms_1} m/s.\nObject 2: {cls_2}, {long_desc_2}, {lat_desc_2}, speed of {speed_ms_2} m/s.\nPlease predict which of the following options best describes Object 1 driving behavior with respect to Object 2."

    def generate_referal(self, man: Maneuver) -> str:
        if man.is_ego and not man.is_other_agent:
            return self.prompt_template.ego_ref + "\n"
        if man.is_ego and man.is_other_agent:
            xy = man.get_other_xys_in_ego()[0]
            cls_2 = man.other_agents[0].general_class_name
            long_desc_2 = (
                f"{abs(int(xy[0]))} meters {'ahead' if xy[0] > 0 else 'behind'}"
            )
            lat_desc_2 = f"{abs(int(xy[1]))} meters {'left' if xy[1] > 0 else 'right'}"
            speed_2 = f"{int(man.other_agents[0].velocity)}"

            return (
                self.prompt_template.ego_ref_other_agent_ref.format(
                    cls_2=cls_2,
                    long_desc_2=long_desc_2,
                    lat_desc_2=lat_desc_2,
                    speed_ms_2=speed_2,
                )
                + "\n"
            )
        elif man.is_agent and not man.is_other_agent:
            xy = man.get_xys_in_ego()[0]
            cls_1 = man.agents[0].general_class_name
            long_desc_1 = (
                f"{abs(int(xy[0]))} meters {'ahead' if xy[0] > 0 else 'behind'}"
            )
            lat_desc_1 = f"{abs(int(xy[1]))} meters {'left' if xy[1] > 0 else 'right'}"
            speed_1 = f"{int(man.agents[0].velocity)}"

            return (
                self.prompt_template.agent_ref.format(
                    cls=cls_1,
                    long_desc=long_desc_1,
                    lat_desc=lat_desc_1,
                    speed_ms=speed_1,
                )
                + "\n"
            )
        elif man.is_agent and man.is_other_agent:
            xy = man.get_xys_in_ego()[0]
            cls_1 = man.agents[0].general_class_name
            long_desc_1 = (
                f"{abs(int(xy[0]))} meters {'ahead' if xy[0] > 0 else 'behind'}"
            )
            lat_desc_1 = f"{abs(int(xy[1]))} meters {'left' if xy[1] > 0 else 'right'}"
            speed_1 = f"{int(man.agents[0].velocity)}"

            xy = man.get_other_xys_in_ego()[0]
            cls_2 = man.other_agents[0].general_class_name
            long_desc_2 = (
                f"{abs(int(xy[0]))} meters {'ahead' if xy[0] > 0 else 'behind'}"
            )
            lat_desc_2 = f"{abs(int(xy[1]))} meters {'left' if xy[1] > 0 else 'right'}"
            speed_2 = f"{int(man.other_agents[0].velocity)}"

            return (
                self.prompt_template.agent_ref_other_agent_ref.format(
                    cls_1=cls_1,
                    long_desc_1=long_desc_1,
                    lat_desc_1=lat_desc_1,
                    speed_ms_1=speed_1,
                    cls_2=cls_2,
                    long_desc_2=long_desc_2,
                    lat_desc_2=lat_desc_2,
                    speed_ms_2=speed_2,
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

    extractor = SennaVQAExtractor()

    cam_order = [
        SensorType.CAM_FRONT,
        SensorType.CAM_FRONT_RIGHT,
        SensorType.CAM_FRONT_LEFT,
        SensorType.CAM_BACK,
        SensorType.CAM_BACK_LEFT,
        SensorType.CAM_BACK_RIGHT,
    ]  # keeps the same cam order as in senna
    qas = []
    for man in tqdm(maneuvers):
        prompt, answer_text, answer_letter = extractor.generate_prompt_answers(man)
        qa = dict(
            images=[
                "data/nuscenes/" + f.get_sensor(c).path
                for f, c in zip(man.frames, cam_order)
            ],
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
