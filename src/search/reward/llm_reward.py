"""Funcitons for atomistic rewards."""
import logging
import sys

import numpy as np

sys.path.append("src")
from llm import query  # noqa: E402
from search.reward.base_reward import BaseReward  # noqa: E402

logging.getLogger().setLevel(logging.INFO)

_default_answer = "answer"


def flatten_prompts(
    prompts: list[list[str]],
) -> tuple[list[tuple[int, int]], list[str]]:
    """Flatten the (jagged) 2D list of prompts and return index mappings."""
    idx = []
    flattend_list = []
    for i in range(len(prompts)):
        for j in range(len(prompts[i])):
            idx.appned((i, j))
            flattend_list.append(prompts[i][j])

    return (idx, flattend_list)


def unflatten_answers(
    idx: list[tuple[int, int]], answers: list[str]
) -> list[list[str]]:
    """Unflatten the given list of answers according to the given (jagged) idx."""
    jagged_answers = [[]]
    loop_counter = 0
    for i, _ in idx:
        if i == len(jagged_answers):
            jagged_answers.append([])
        jagged_answers[i].append(answers[loop_counter])
        loop_counter += 1
    return jagged_answers


class LLMRewardFunction(BaseReward):
    """A class for the llm reward function."""

    def __init__(
        self,
        llm_function: callable,
        reward_limit: float = 10.0,
        max_attempts: int = 3,
        penalty_value: float = -10.0,
    ):
        """Create the LLMRewardFunction from given parameters."""
        self.llm_function = llm_function
        self.reward_limit = reward_limit
        self.max_attempts = max_attempts
        self.penalty_reward = penalty_value

    def __call__(
        self,
        states: list[query.QueryState],
        primary_reward=True,
    ):
        """Reward function to return adsorption energy of reactants.

        The calculated reward is stored in state.reward.
        """
        rewards = [None] * len(states)
        attempts = 0
        while any([r is None for r in rewards]) or attempts < self.max_attempts:
            prompts = []
            system_prompts = []
            for i, s in enumerate(states):
                if rewards[i] is not None:
                    prompts.append(s.generation_prompt)
                    system_prompts.append(s.system_prompt_generation)

            flatten_idx, flattened_prompts = flatten_prompts(prompts)

            answers = self.llm_function(flattened_prompts)

            unflattened_answers = unflatten_answers(flatten_idx, answers)

            loop_counter = 0
            for i, s in enumerate(states):
                if rewards[i] is None:
                    try:
                        values = s.process_adsorption_energy(
                            unflattened_answers[loop_counter]
                        )
                        reward = np.mean(
                            [
                                np.mean(cat_values) ** (s.ads_preferences[j])
                                for j, cat_values in enumerate(values)
                            ]
                        )
                        rewards[i] = reward

                    except Exception as err:
                        if attempts < self.max_attempts - 1:
                            logging.warning(
                                f"Failed to parse answer with error: {str(err)}. "
                                "Generating new answer."
                            )
                        else:
                            logging.warning(
                                f"Failed to parse answer with error: {str(err)}. "
                                "Returning a penalty value."
                            )
                            rewards[i] = self.penalty_value
                    loop_counter += 1
            attempts += 1
        # end while

        # one last loop to save all the values
        for i, s in enumerate(states):
            s.set_reward(
                rewards[i], info_field="llm-reward", primary_reward=self.primary_reward
            )

        return rewards
