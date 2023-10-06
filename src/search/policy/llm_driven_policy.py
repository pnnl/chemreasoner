"""Class for the coherence policy."""
import sys

from collections.abc import Callable

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

sys.path.append("src")
from search.policy.reasoner_policy import ReasonerPolicy  # noqa:402


class LLMDrivenPolicy(ReasonerPolicy):
    """A polocy like the Reasoner policy, but it promotes more coherent prompts."""

    def __init__(
        self,
        temperature: float = 0.6,
        include_property_types: list[str] = None,
        exclude_property_types: list[str] = None,
        relationship_to_candidate_list_types: list[str] = None,
        catalyst_label_types: list[str] = None,
        try_oxides: bool = True,
    ):
        """Create the underlying ReasonerPolicy."""
        super().__init__(
            include_property_types,
            exclude_property_types,
            relationship_to_candidate_list_types,
            catalyst_label_types,
            try_oxides,
        )
        self.temperature = temperature

    @classmethod
    @staticmethod
    def from_reasoner_policy(
        reasoner_policy: ReasonerPolicy, temperature: float = 0.6
    ) -> "LLMDrivenPolicy":
        """Construct a coherent policy from a reasoner poliy."""
        p = LLMDrivenPolicy()
        p.actions = reasoner_policy.actions.copy()
        p.init_weights()
        return p

    def get_actions(
        self, state: object
    ) -> tuple[list[Callable[object, object]], np.array]:
        """Return the actions along with their priors."""
        actions, priors = super().get_actions(
            state
        )  # get super class actions and priors

        prev_answer = state.answer

        prior_prompt = f"Consider the previous answer: {prev_answer}.\n\n"
        prior_prompt += (
            "Your task is to rate the following actions to produce a "
            "new prompt that an llm can use to recommend better catalysts.\n\n"
        )

        actions_statement = "The actions are:\n\n"
        for a, i in enumerate(actions):
            actions_statement += f"{i}) {a.message}\n\n"  # punctuation is in a.message

        prior_prompt += actions_statement
        prior_prompt += (
            "Return your answer as a python dictionary mapping each "
            "action to a score from 0 to 10 (10 is the best)."
        )

        return actions, new_priors


if __name__ == "__main__":
    from llm.automate_prompts import get_initial_state_oc

    s, _ = get_initial_state_oc("H20", "gpt-3.5-turbo", "gpt-3.5-turbo")
    p = LLMDrivenPolicy(0.4)
    print(p.temperature)
    print(p.get_actions(s))
