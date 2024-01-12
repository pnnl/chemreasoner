"""Class for the coherence policy."""
import logging
import sys
import time

from collections.abc import Callable

import numpy as np

sys.path.append("src")
from search.policy.reasoner_policy import (  # noqa:402
    IncludePropertyAdder,
    ExcludePropertyAdder,
    RelationToCandidateListChanger,
)
from search.policy.policy_base import BasePolicy  # noqa:402
from search.state.reasoner_state import ReasonerState  # noqa:402


class CatalystLabelChanger:
    """Class to change catalyst label of a state."""

    def __init__(self, catalyst_label_type):
        """Save the property name."""
        self.catalyst_label_type = catalyst_label_type

        self._message = f"Predict {catalyst_label_type}."

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        new_state.catalyst_label = self.catalyst_label_type
        return new_state

    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message


logging.getLogger().setLevel(logging.INFO)

action_name_keys = {
    "catalyst_type": CatalystLabelChanger,
    "inclusion_criteria": IncludePropertyAdder,
    "exclusion_criteria": ExcludePropertyAdder,
    "relationship_to_candidate_list": RelationToCandidateListChanger,
}

example_output = """
Therefore, the suggested actions to narrow down the search and accomplish the $root_prompt are:
{
   "catalyst_type": ["bimetallic catalysts", "transition metal catalysts"],
   "inclusion_criteria": ["high selectivity", "high producability],
   "exclusion_criteria": ["poor stability", ""],
   "relationship_to_candidate_list": ["complementary to"]
}
"""


priors_template = """$search_state = {current_state}

$action_space = {action_space}

$root_question = {root_prompt}

{current_prompt_answer}
Consider the {current_conditions}. Your task is to suggest possible actions that could achieve the intent of the $root_prompt. 

Your answers should use the following guidelines:
{guidelines}

{final_task}
"""


class CoherentPolicy(BasePolicy):
    """A polocy like the Reasoner policy, but it promotes more coherent prompts."""

    def __init__(
        self,
        llm_function: callable = lambda list_x: [example_output] * len(list_x),
        max_num_actions: int = 10,
        max_attempts: int = 3,
    ):
        """Create the underlying ReasonerPolicy."""
        self.max_num_actions = max_num_actions
        self.llm_function = llm_function
        self.max_attempts = max_attempts

    @staticmethod
    def strings_to_actions(action_lists: dict[str, str]) -> list[callable]:
        """Turn the strings returned by the language model into actions."""
        actions = []
        for k, v in action_lists.items():
            actions += [action_name_keys[k](a) for a in v]
        return actions

    def get_actions(
        self,
        states: list[ReasonerState],
    ) -> tuple[list[Callable], np.array]:
        """Return the actions along with their priors."""
        attempts = 0
        action_priors = [None] * len(states)
        start = time.time()
        while any([i is None for i in action_priors]) and attempts < self.max_attempts:
            attempts += 1
            prompts = []
            prompts_idx = []
            for i, s in enumerate(states):
                if s.priors_template is None:
                    s.priors_template = priors_template
                try:
                    prompts.append(s.priors_prompt)
                    prompts_idx.append(i)
                except Exception:
                    logging.warning("Cannot generate prompt for state.")

            if len(prompts) > 0:
                llm_answers = self.llm_function(prompts)

                for i, ans in enumerate(llm_answers):
                    try:
                        s = states[prompts_idx[i]]
                        action_lists = s.process_prior(ans)
                        actions = self.strings_to_actions(action_lists)

                        if len(actions) >= self.max_num_actions:
                            actions = actions[: self.max_num_actions]
                            priors = np.array([1 / len(actions)] * len(actions))
                        elif len(actions) < self.max_num_actions:
                            length_difference = self.max_num_actions - len(actions)
                            priors = np.array(
                                [1 / len(actions)] * len(actions)
                                + [0] * length_difference
                            )
                            actions += [None] * length_difference

                        action_priors[prompts_idx[i]] = (actions, priors)
                    except Exception:
                        logging.warning(
                            "Could not parse the actions for the given state. Trying again."
                        )
        end = time.time()
        logging.info(f"TIMING: Get actions time {end-start}")
        action_priors = [a_p if a_p is not None else [] for a_p in action_priors]

        return action_priors


if __name__ == "__main__":
    import pickle

    p = CoherentPolicy(max_num_actions=5)

    with open("data/example_trajectory.pkl", "rb") as f:
        states = pickle.load(f)
    for i, s in enumerate(states):
        if i == 0:
            root_prompt = s.generation_prompt
        s.info.pop("priors")
        dict_data = vars(s)
        dict_data.update(
            {"priors_template": priors_template, "root_prompt": root_prompt}
        )

    print(states)
    print(p.get_actions(states))
