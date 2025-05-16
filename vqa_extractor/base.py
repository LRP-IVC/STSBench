from typing import List, Tuple
from easydict import EasyDict
import random
import string
from abc import ABC, abstractmethod

import numpy as np

from annotator.data.maneuvers import Maneuver, ManeuverType
from annotator.mining.consts import (
    MANEUVER_DESCRIPTION,
    short_maneuver_description,
    long_maneuver_description,
)


class VQAExtractor(ABC):
    def __init__(
        self,
        number_of_negatives: int = 4,
        ego_desc="ego",
        agent_desc="agent",
        other_agent_desc="agent",
        short_desc_generator=short_maneuver_description,
        long_desc_generator=long_maneuver_description,
        prompt_template: dict = dict(
            preamble="""You are a traffic control expert specializing in the analysis and identification of temporal actions and maneuvers performed by various agents, as well as the ego vehicle, across diverse driving scenarios.

    Agents refer to all participants in the traffic environment, including but not limited to: cars, buses, construction vehicles, trucks, trailers, motorcycles, pedestrians, and bicycles.
    Ego Vehicle is the primary vehicle from whose perspective the scenario is being evaluated. It is equipped with a sophisticated suite of sensors (e.g., cameras, LiDAR, radar) to capture the surrounding traffic environment.
    Temporal Actions and Maneuvers include any time-based behaviors or movements, such as lane changes, accelerations, decelerations, turns, stops, or interactions between agents (including the ego).

    Your expertise involves interpreting complex traffic dynamics, identifying patterns, and assessing the implications of these actions within the context of real-world driving scenarios. Your goal is to provide detailed, accurate, and actionable insights into the behavior of all agents, including the ego vehicle, to enhance traffic safety and efficiency.""",
            maneuver_description="""The following are driving maneuvers and actions along with their respective descriptions:
    {man_desc}""",
            omnidrive_ref="If you observe the object at the location ({x:+.2f}, {y:+.2f}), which of the following options best describes the object's driving maneuver?",
            ego_ref="Which of the following options best describes your driving maneuver?",
            options="Options:\n{answers}",
            postamble="""Respond with only the letter corresponding to your choice (e.g., 'A', 'B', etc.) and nothing else.""",
        ),
    ) -> None:
        self.number_of_negatives = number_of_negatives
        self.ego_desc = ego_desc
        self.agent_desc = agent_desc
        self.other_agent_desc = other_agent_desc
        self.prompt_template = EasyDict(prompt_template)
        self.short_desc_generator = short_desc_generator
        self.long_desc_generator = long_desc_generator

    def generate_prompt_answers(self, man: Maneuver):
        assert len(man.pos_maneuvers) > 0, "No positive maneuver"
        assert len(man.neg_maneuvers) >= self.number_of_negatives, (
            f"Not enough negative maneuvers {len(man.neg_maneuvers)}"
        )
        multiple_choice, answer_text, answer_letter, subsampled_man_types = self.generate_multiple_choice(man)
        prompt = ""
        prompt += self.generate_preamble(man)
        prompt += self.generate_man_description(man, subsampled_man_types)
        prompt += self.generate_referal(man)
        prompt += multiple_choice
        prompt += self.generate_postamble(man)
        return prompt, answer_text, answer_letter

    def generate_preamble(self, man: Maneuver) -> str:
        return self.prompt_template.preamble + "\n"

    def generate_man_description(self, man: Maneuver, man_types: List[ManeuverType]) -> str:
        man_desc = "\n"
        for man_type in man_types:
            short_description = self.short_desc_generator(
                man_type,
                man.is_ego,
                man.is_agent,
                ego_desc=self.ego_desc,
                agent_desc=self.agent_desc,
                other_agent_desc=self.other_agent_desc,
            )
            long_description = self.long_desc_generator(
                man_type,
                man.is_ego,
                man.is_agent,
                ego_desc=self.ego_desc,
                agent_desc=self.agent_desc,
                other_agent_desc=self.other_agent_desc,
            )
            man_desc += short_description + ": " + long_description + "\n"
        return (
            self.prompt_template.maneuver_description.format(man_desc=man_desc)
        )

    @abstractmethod
    def generate_referal(self, man: Maneuver) -> str:
        pass

    def generate_multiple_choice(self, man: Maneuver) -> Tuple[str, str, str, List[ManeuverType]]:
        # sample from negatives
        neg_types = [n.type for n in man.neg_maneuvers]
        neg_types_sampled = self._linear_weighted_sample(
            neg_types, self.number_of_negatives
        )

        # create multiple choice
        all_types = [man.pos_maneuvers[0].type] + neg_types_sampled
        random.shuffle(all_types)
        correct_id = all_types.index(man.pos_maneuvers[0].type)
        multiple_choice = ""
        for letter, answer in zip(string.ascii_uppercase, all_types):
            short_description = self.short_desc_generator(
                answer,
                man.is_ego,
                man.is_agent,
                ego_desc=self.ego_desc,
                agent_desc=self.agent_desc,
                other_agent_desc=self.other_agent_desc,
            )
            multiple_choice += f"{letter}. {short_description}\n"
        # remove last \n
        multiple_choice.rstrip()

        answer_text = self.short_desc_generator(
            man.pos_maneuvers[0].type,
            man.is_ego,
            man.is_agent,
            ego_desc=self.ego_desc,
            agent_desc=self.agent_desc,
            other_agent_desc=self.other_agent_desc,
        )
        answer_letter = string.ascii_uppercase[correct_id]

        return (
            self.prompt_template.options.format(answers=multiple_choice),
            answer_text,
            answer_letter,
            all_types,
        )

    def _linear_weighted_sample(self, lst: List, k: int = 1) -> List:
        n = len(lst)
        if n == 0:
            return []
        weights = np.array([n - i for i in range(n)])
        probabilities = weights / weights.sum()
        indices = np.random.choice(n, size=k, replace=False, p=probabilities)
        return [lst[i] for i in indices]

    def generate_postamble(self, man: Maneuver) -> str:
        return self.prompt_template.postamble
