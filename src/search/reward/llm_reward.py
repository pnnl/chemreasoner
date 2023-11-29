"""Funcitons for atomistic rewards."""
import logging
import sys

import numpy as np

sys.path.append("src")
from search.state.reasoner_state import ReasonerState  # noqa: E402
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
            idx.append((i, j))
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
        self.penalty_value = penalty_value

    def run_generation_promtps_openai(self):
        """Run the generation and adsorbate prompts in one go with function calling.

        https://platform.openai.com/docs/guides/function-calling
        """
        ...

    def run_generation_prompts(self, rewards, states):
        """Run the generation prompts for the given states where the reward is None."""
        prompts = []
        system_prompts = []
        for i, s in enumerate(states):
            if rewards[i] is None:
                prompts.append(s.generation_prompt)
                system_prompts.append(s.generation_system_prompt)

        generation_results = self.llm_function(prompts, system_prompts)
        loop_counter = 0
        for i, s in enumerate(states):
            if rewards[i] is None:
                s.process_generation(generation_results[loop_counter])

                loop_counter += 1

    def run_adsorption_energy_prompts(self, rewards, states):
        """Run the generation prompts for the given states where the reward is None."""
        prompts = []
        system_prompts = []
        prompts_idx = []
        for i, s in enumerate(states):
            if rewards[i] is None:
                try:
                    prompts.append(s.adsorption_energy_prompts)
                    system_prompts.append(s.reward_system_prompt)
                    prompts_idx.append(i)
                except Exception as err:
                    logging.warning(
                        f"Failed to generate prompts with error: {str(err)}. "
                        "Skipping this prompt."
                    )
                    if len(prompts) > len(system_prompts):
                        prompts.pop()

        flatten_idx, flattened_prompts = flatten_prompts(prompts)
        _, flattened_system_prompts = flatten_prompts(prompts)

        flattened_results = self.llm_function(
            flattened_prompts, flattened_system_prompts
        )

        answers = unflatten_answers(flatten_idx, flattened_results)
        for i, p in enumerate(prompts):
            state_idx = prompts_idx[i]
            s = states[state_idx]

            try:
                values = s.process_adsorption_energy(answers[i])
                reward = np.mean(
                    [
                        np.mean(cat_values) ** (s.ads_preferences[j])
                        for j, cat_values in enumerate(values)
                    ]
                )
                rewards[state_idx] = reward

            except Exception as err:
                logging.warning(f"Failed to parse answer with error: {str(err)}.")

    def __call__(
        self,
        states: list[ReasonerState],
        primary_reward=True,
    ):
        """Reward function to return adsorption energy of reactants.

        The calculated reward is stored in state.reward.
        """
        rewards = [None] * len(states)
        attempts = 0
        while any([r is None for r in rewards]) and attempts < self.max_attempts:
            if primary_reward:
                self.run_generation_prompts(rewards, states)

            self.run_adsorption_energy_prompts(rewards, states)

            attempts += 1

        rewards = [r if r is not None else self.penalty_value for r in rewards]
        # end while

        # one last loop to save all the values
        for i, s in enumerate(states):
            s.set_reward(
                rewards[i], info_field="llm-reward", primary_reward=primary_reward
            )

        return rewards
