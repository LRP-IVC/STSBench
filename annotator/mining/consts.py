from ..data.maneuvers import ManeuverType


def short_maneuver_description(
    man_type,
    is_ego,
    is_agent,
    ego_desc="ego",
    agent_desc="agent",
    other_agent_desc="agent",
) -> str:
    if man_type == ManeuverType.ACCELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} is accelerating"
        elif is_agent:
            return f"{agent_desc.capitalize()} is accelerating"
        else:
            assert False
    elif man_type == ManeuverType.DECELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} is decelerating"
        elif is_agent:
            return f"{agent_desc.capitalize()} is decelerating"
        else:
            assert False
    elif man_type == ManeuverType.LANE_CHANGE:
        if is_ego:
            return f"{ego_desc.capitalize()} is changing lanes"
        elif is_agent:
            return f"{agent_desc.capitalize()} is changing lanes"
        else:
            assert False
    elif man_type == ManeuverType.LEFT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is turning left"
        elif is_agent:
            return f"{agent_desc.capitalize()} is turning left"
        else:
            assert False
    elif man_type == ManeuverType.RIGHT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is turning right"
        elif is_agent:
            return f"{agent_desc.capitalize()} is turning right"
        else:
            assert False
    elif man_type == ManeuverType.U_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is performing u-turn"
        elif is_agent:
            return f"{agent_desc.capitalize()} is performing u-turn"
        else:
            assert False
    elif man_type == ManeuverType.REVERSE:
        if is_ego:
            return f"{ego_desc.capitalize()} is reversing"
        elif is_agent:
            return f"{agent_desc.capitalize()} is reversing"
        else:
            assert False
    elif man_type == ManeuverType.STOP:
        if is_ego:
            return f"{ego_desc.capitalize()} is stopping"
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
        return f"{agent_desc} is passing stationary {ego_desc}".capitalize()
    elif man_type == ManeuverType.OVERTAKE_AGENT:
        if is_ego:
            return f"{ego_desc} is overtaking {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is overtaking {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.WAIT_PED_CROSS:
        if is_ego:
            return f"{ego_desc} is waiting for pedestrian to cross".capitalize()
        elif is_agent:
            return f"{agent_desc} is waiting for pedestrian to cross".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.FOLLOW_AGENT:
        if is_ego:
            return f"{ego_desc} is following {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is following {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.LEAD_AGENT:
        if is_ego:
            return f"{ego_desc} is leading {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is leading {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.PASS_AGENT:
        if is_ego:
            return f"{ego_desc} is passes stationary {other_agent_desc}".capitalize()
        elif is_agent:
            return f"{agent_desc} is passes stationary {other_agent_desc}".capitalize()
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
            return f"{agent_desc} is standing".capitalize()
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
            return f"{ego_desc} is stationary behind {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary in front of {other_agent_desc}".capitalize()
        elif is_ego:
            return (
                f"{ego_desc} is stationary in front of {other_agent_desc}".capitalize()
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
            return f"{agent_desc} is stationary in front of {ego_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_RIGHT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary to the right of {other_agent_desc}".capitalize()
        elif is_ego:
            return f"{ego_desc} is stationary to the right of {other_agent_desc}".capitalize()
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc} is stationary to the left of {other_agent_desc}".capitalize()
        elif is_ego:
            return f"{ego_desc} is stationary to the left of {other_agent_desc}".capitalize()
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
                f"{ego_desc} is moving to the right of {other_agent_desc}".capitalize()
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
                f"{ego_desc} is moving to the left of {other_agent_desc}".capitalize()
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
    ego_desc="ego",
    agent_desc="agent",
    other_agent_desc="agent",
) -> str:
    if man_type == ManeuverType.ACCELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} is increasing its speed, either gradually or abruptly, to adapt to traffic conditions, maintain flow, or comply with traffic rules and signals."
        elif is_agent:
            return f"{agent_desc.capitalize()} is increasing its speed, either gradually or abruptly, to adapt to traffic conditions, maintain flow, or comply with traffic rules and signals."
        else:
            assert False
    elif man_type == ManeuverType.DECELERATE:
        if is_ego:
            return f"{ego_desc.capitalize()} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules, without coming to a complete stop."
        elif is_agent:
            return f"{agent_desc.capitalize()} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules, without coming to a complete stop."
        else:
            assert False
    elif man_type == ManeuverType.LANE_CHANGE:
        if is_ego:
            return f"{ego_desc.capitalize()} is transitioning from its current lane to an adjacent lane."
        elif is_agent:
            return f"{agent_desc.capitalize()} is transitioning from its current lane to an adjacent lane."
        else:
            assert False
    elif man_type == ManeuverType.LEFT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is executing a left turn at an intersection or junction."
        elif is_agent:
            return f"{agent_desc.capitalize()} is executing a left turn at an intersection or junction."
        else:
            assert False
    elif man_type == ManeuverType.RIGHT_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is executing a right turn at an intersection or junction."
        elif is_agent:
            return f"{agent_desc.capitalize()} is executing a right turn at an intersection or junction."
        else:
            assert False
    elif man_type == ManeuverType.U_TURN:
        if is_ego:
            return f"{ego_desc.capitalize()} is performing a 180-degree turn at an intersection or junction, reversing its direction of travel."
        elif is_agent:
            return f"{agent_desc.capitalize()} is performing a 180-degree turn at an intersection or junction, reversing its direction of travel."
        else:
            assert False
    elif man_type == ManeuverType.REVERSE:
        if is_ego:
            return f"{ego_desc.capitalize()} is moving in reverse, either to park, navigate a tight space, or adjust its position."
        elif is_agent:
            return f"{agent_desc.capitalize()} is moving in reverse, either to park, navigate a tight space, or adjust its position."
        else:
            assert False
    elif man_type == ManeuverType.STOP:
        if is_ego:
            return f"{ego_desc.capitalize()} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules and comes to a complete stop."
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
            return f"{ego_desc.capitalize()} in the adjacent lane and moves ahead of {other_agent_desc} while both are in motion."
        elif is_agent:
            return f"{agent_desc.capitalize()} in the adjacent lane and moves ahead of {other_agent_desc} while both are in motion."
        else:
            assert False
    elif man_type == ManeuverType.WAIT_PED_CROSS:
        if is_ego:
            return f"{ego_desc.capitalize()} comes to a stop or remains stationary, yielding the right-of-way to {other_agent_desc} who is crossing or preparing to cross the road, while maintaining awareness of the {other_agent_desc}'s movement and ensuring a safe distance until the crossing is complete."
        elif is_agent:
            return f"{agent_desc.capitalize()} comes to a stop or remains stationary, yielding the right-of-way to a {other_agent_desc} who is crossing or preparing to cross the road, while maintaining awareness of the {other_agent_desc}'s movement and ensuring a safe distance until the crossing is complete."
        else:
            assert False
    elif man_type == ManeuverType.FOLLOW_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} is driving behind {other_agent_desc} at a similar speed while maintaining a consistent distance."
        elif is_agent:
            return f"{agent_desc.capitalize()} is driving behind {other_agent_desc} at a similar speed while maintaining a consistent distance."
        else:
            assert False
    elif man_type == ManeuverType.LEAD_AGENT:
        if is_ego:
            return f"{ego_desc.capitalize()} travels ahead of {other_agent_desc} at a similar speed while maintaining a consistent distance."
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
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary behind {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_IN_FRONT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary in front of {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
        elif is_ego:
            return f"{ego_desc.capitalize()} is fully stopped and remains stationary in front of {other_agent_desc}, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing."
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
            return f"{ego_desc.capitalize()} is fully stopped and remains stationary to the right of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        else:
            assert False
    elif man_type == ManeuverType.STATIONARY_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is fully stopped and remains stationary to the left of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
        elif is_ego:
            return f"{ego_desc.capitalize()} is fully stopped and remains stationary to the left of {other_agent_desc}, which is also stationary, such as when waiting at a traffic light or in a parking lot."
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
            return f"{ego_desc.capitalize()} is traveling in parallel to the right of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        else:
            assert False
    elif man_type == ManeuverType.MOVING_LEFT_OF_AGENT:
        if is_agent:
            return f"{agent_desc.capitalize()} is traveling in parallel to the left of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
        elif is_ego:
            return f"{ego_desc.capitalize()} is traveling in parallel to the left of {other_agent_desc} (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light."
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


NEGATIVE_MANEUVERS = {
    ManeuverType.ACCELERATE: [
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.DECELERATE,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.DECELERATE: [
        ManeuverType.STOP,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.LANE_CHANGE,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.WAIT_PED_CROSS: [
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.DECELERATE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.STOP,
        ManeuverType.REVERSE,
        ManeuverType.U_TURN,
    ],
    ManeuverType.STOP: [
        ManeuverType.DECELERATE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.LANE_CHANGE,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.LANE_CHANGE: [
        ManeuverType.DECELERATE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.STOP,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.LEFT_TURN: [
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.STOP,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.RIGHT_TURN: [
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.STOP,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.U_TURN: [
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.STOP,
        ManeuverType.REVERSE,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.REVERSE: [
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.STOP,
        ManeuverType.U_TURN,
        ManeuverType.ACCELERATE,
    ],
    ManeuverType.LEAD_EGO: [
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.FOLLOW_EGO: [
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.OVERTAKE_EGO: [
        ManeuverType.PASS_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.PASS_EGO: [
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.STATIONARY_BEHIND_EGO: [
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.STATIONARY_IN_FRONT_OF_EGO: [
        ManeuverType.LEAD_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.STATIONARY_RIGHT_OF_EGO: [
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.STATIONARY_LEFT_OF_EGO: [
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.MOVING_RIGHT_OF_EGO: [
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.MOVING_LEFT_OF_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.MOVING_LEFT_OF_EGO: [
        ManeuverType.OVERTAKE_EGO,
        ManeuverType.PASS_EGO,
        ManeuverType.STATIONARY_LEFT_OF_EGO,
        ManeuverType.MOVING_RIGHT_OF_EGO,
        ManeuverType.STATIONARY_RIGHT_OF_EGO,
        ManeuverType.FOLLOW_EGO,
        ManeuverType.LEAD_EGO,
        ManeuverType.STATIONARY_IN_FRONT_OF_EGO,
        ManeuverType.STATIONARY_BEHIND_EGO,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.LANE_CHANGE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.STOP,
    ],
    ManeuverType.OVERTAKE_AGENT: [
        ManeuverType.PASS_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.PASS_AGENT: [
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.FOLLOW_AGENT: [
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.LEAD_AGENT: [
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.STATIONARY_BEHIND_AGENT: [
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.STATIONARY_IN_FRONT_OF_AGENT: [
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.STATIONARY_RIGHT_OF_AGENT: [
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.STATIONARY_LEFT_OF_AGENT: [
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.MOVING_RIGHT_OF_AGENT: [
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.MOVING_LEFT_OF_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.MOVING_LEFT_OF_AGENT: [
        ManeuverType.PASS_AGENT,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.STATIONARY_LEFT_OF_AGENT,
        ManeuverType.MOVING_RIGHT_OF_AGENT,
        ManeuverType.STATIONARY_RIGHT_OF_AGENT,
        ManeuverType.LEAD_AGENT,
        ManeuverType.FOLLOW_AGENT,
        ManeuverType.STATIONARY_IN_FRONT_OF_AGENT,
        ManeuverType.STATIONARY_BEHIND_AGENT,
        ManeuverType.WAIT_PED_CROSS,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
        ManeuverType.LANE_CHANGE,
        ManeuverType.U_TURN,
        ManeuverType.REVERSE,
        ManeuverType.LEFT_TURN,
        ManeuverType.RIGHT_TURN,
    ],
    ManeuverType.CROSS: [
        ManeuverType.JAYWALK,
        ManeuverType.RUN,
        ManeuverType.WALK,
        ManeuverType.STAND,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
    ],
    ManeuverType.JAYWALK: [
        ManeuverType.CROSS,
        ManeuverType.RUN,
        ManeuverType.WALK,
        ManeuverType.STAND,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
    ],
    ManeuverType.RUN: [
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
        ManeuverType.WALK,
        ManeuverType.STAND,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
    ],
    ManeuverType.STAND: [
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
        ManeuverType.WALK,
        ManeuverType.RUN,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
    ],
    ManeuverType.WALK: [
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
        ManeuverType.STAND,
        ManeuverType.RUN,
        ManeuverType.ACCELERATE,
        ManeuverType.DECELERATE,
        ManeuverType.STOP,
    ],
    ManeuverType.WALK_ALONGSIDE: [
        ManeuverType.WALK_OPPOSITE,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.RUN,
        ManeuverType.STAND,
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
    ],
    ManeuverType.WALK_OPPOSITE: [
        ManeuverType.WALK_ALONGSIDE,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.PASS_AGENT,
        ManeuverType.RUN,
        ManeuverType.STAND,
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
    ],
}

PED_NEGATIVE_MANEUVERS = {
    ManeuverType.OVERTAKE_AGENT: [
        ManeuverType.WALK_ALONGSIDE,
        ManeuverType.WALK_OPPOSITE,
        ManeuverType.PASS_AGENT,
        ManeuverType.WALK,
        ManeuverType.RUN,
        ManeuverType.STAND,
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
    ],
    ManeuverType.PASS_AGENT: [
        ManeuverType.WALK_ALONGSIDE,
        ManeuverType.WALK_OPPOSITE,
        ManeuverType.OVERTAKE_AGENT,
        ManeuverType.WALK,
        ManeuverType.RUN,
        ManeuverType.STAND,
        ManeuverType.CROSS,
        ManeuverType.JAYWALK,
    ],
}

MANEUVER_DESCRIPTION = {
    ManeuverType.ACCELERATE: "Accelerating: {traffic_participant} is increasing its speed, either gradually or abruptly, to adapt to traffic conditions, maintain flow, or comply with traffic rules and signals.",
    ManeuverType.DECELERATE: "Decelerating: {traffic_participant} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules, without coming to a complete stop.",
    ManeuverType.LANE_CHANGE: "Lane change: {traffic_participant} is transitioning from its current lane to an adjacent lane.",
    ManeuverType.LEFT_TURN: "Turning left: {traffic_participant} is executing a left turn at an intersection or junction.",
    ManeuverType.RIGHT_TURN: "Turning right: {traffic_participant} is executing a right turn at an intersection or junction.",
    ManeuverType.U_TURN: "U-turn: {traffic_participant} is performing a 180-degree turn at an intersection or junction, reversing its direction of travel.",
    ManeuverType.REVERSE: "Reversing: {traffic_participant} is moving in reverse, either to park, navigate a tight space, or adjust its position.",
    ManeuverType.STOP: "Stopping: {traffic_participant} is reducing its speed, either gradually or abruptly, in response to traffic conditions, obstacles, or to comply with traffic rules and comes to a complete stop.",
    ManeuverType.OVERTAKE_EGO: "Agent overtaking ego: Agent in the adjacent lane moves ahead of the ego vehicle while both are in motion.",
    ManeuverType.FOLLOW_EGO: "Agent following ego: Agent is driving behind the ego vehicle at a similar speed while maintaining a consistent distance.",
    ManeuverType.LEAD_EGO: "Agent leading ego: Agent travels ahead of the ego vehicle at a similar speed while maintaining a consistent distance.",
    ManeuverType.OVERTAKE_AGENT: "{traffic_participant} overtaking agent: {traffic_participant} in the adjacent lane moves ahead of the agent while both are in motion.",
    ManeuverType.WAIT_PED_CROSS: "{traffic_participant} waiting for pedestrian to cross: {traffic_participant} comes to a stop or remains stationary, yielding the right-of-way to a pedestrian who is crossing or preparing to cross the road, while maintaining awareness of the pedestrian's movement and ensuring a safe distance until the crossing is complete.",
    ManeuverType.FOLLOW_AGENT: "{traffic_participant} following agent: {traffic_participant} is driving behind the agent at a similar speed while maintaining a consistent distance.",
    ManeuverType.LEAD_AGENT: "{traffic_participant} leading agent: {traffic_participant} travels ahead of the agent at a similar speed while maintaining a consistent distance.",
    ManeuverType.PASS_AGENT: "{traffic_participant} passes stationary agent: {traffic_participant} in the adjacent lane overtakes the stopped agent.",
    ManeuverType.PASS_EGO: "Agent passes stationary ego vehicle: Agent in the adjacent lane overtakes the stopped ego vehicle.",
    ManeuverType.CROSS: "Agent crossing street: Agent (pedestrian) moves from one side of the road to the other, at a designated crossing point or intersection.",
    ManeuverType.JAYWALK: "Agent jaywalking: Agent (pedestrian) crosses the street outside of designated crossing areas or against traffic signals, often requiring heightened awareness of vehicle movements, quick decision-making to avoid conflicts, and potentially creating unpredictable interactions with other agents in the traffic environment.",
    ManeuverType.RUN: "Agent running: Agent (pedestrian) moves rapidly.",
    ManeuverType.WALK: "Agent walking: Agent (pedestrian) moves at a steady, moderate pace, typically following designated paths or crosswalks.",
    ManeuverType.STAND: "Agent standing: Agent (pedestrian) remains stationary in the traffic environment, either waiting at a crossing, observing surroundings, or pausing for other reasons.",
    ManeuverType.WALK_ALONGSIDE: "Agents walking together: Two pedestrians walk side by side at a steady, moderate pace.",
    ManeuverType.WALK_OPPOSITE: "Agents walking in opposite directions: The two agents (pedestrians) walk toward each other at a moderate pace, cross paths, and proceed.",
    ManeuverType.STATIONARY_BEHIND_AGENT: "{traffic_participant} stationary behind stationary agent: {traffic_participant} is fully stopped and remains stationary behind another stationary vehicle, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing.",
    ManeuverType.STATIONARY_IN_FRONT_OF_AGENT: "{traffic_participant} stationary in front of stationary agent: {traffic_participant} is fully stopped and remains stationary ahead of another stationary vehicle, such as when waiting at a traffic light, in a parking lot, or any other situation requiring queuing.",
    ManeuverType.STATIONARY_BEHIND_EGO: "Agent stationary behind stationary ego: A vehicle is fully stopped and remains stationary behind the ego vehicle (which is also stopped), such as when waiting at a traffic light, in a parking lot, or in any other queuing scenario.",
    ManeuverType.STATIONARY_IN_FRONT_OF_EGO: "Agent stationary in front of stationary ego: A vehicle is fully stopped and remains stationary ahead of the ego vehicle (which is also stopped), such as when waiting at a traffic light, in a parking lot, or in any other queuing scenario.",
    ManeuverType.STATIONARY_RIGHT_OF_AGENT: "{traffic_participant} stationary to the right of stationary agent: {traffic_participant} is fully stopped and remains stationary to the right of another stationary vehicle, such as when waiting at a traffic light or in a parking lot.",
    ManeuverType.STATIONARY_LEFT_OF_AGENT: "{traffic_participant} stationary to the left of stationary agent: {traffic_participant} is fully stopped and remains stationary to the left of another stationary vehicle, such as when waiting at a traffic light or in a parking lot.",
    ManeuverType.STATIONARY_RIGHT_OF_EGO: "Agent stationary to the right of stationary ego: A vehicle is fully stopped and remains stationary to the right of the ego vehicle (which is also stopped), such as when waiting at a traffic light or in a parking lot.",
    ManeuverType.STATIONARY_LEFT_OF_EGO: "Agent stationary to the left of stationary ego: A vehicle is fully stopped and remains stationary to the left of the ego vehicle (which is also stopped), such as when waiting at a traffic light or in a parking lot.",
    ManeuverType.MOVING_RIGHT_OF_AGENT: "{traffic_participant} is moving to the right of the moving agent: Two vehicles are traveling in parallel (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light.",
    ManeuverType.MOVING_LEFT_OF_AGENT: "{traffic_participant} is moving to the left of the moving agent: Two vehicles are traveling in parallel (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light.",
    ManeuverType.MOVING_RIGHT_OF_EGO: "Agent is moving to the right of the moving ego vehicle: Agent and ego vehicles are traveling in parallel (e.g., in adjacent lanes or side by side), with one vehicle maintaining a rightward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light.",
    ManeuverType.MOVING_LEFT_OF_EGO: "Agent is moving to the left of the moving ego vehicle: Agent and ego vehicles are traveling in parallel (e.g., in adjacent lanes or side by side), with one vehicle maintaining a leftward offset relative to the other. This could occur during lane-matched driving on a multi-lane road or synchronized movement from a traffic light.",
}
