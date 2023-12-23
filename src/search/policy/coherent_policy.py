"""Class for the coherence policy."""
import sys

from collections.abc import Callable

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

sys.path.append("src")
from search.policy.reasoner_policy import (  # noqa:402
    CatalystLabelChanger,
    IncludePropertyAdder,
    ExcludePropertyAdder,
    RelationToCandidateListChanger,
)
from search.policy.base_policy import BasePolicy  # noqa:402
from search.state.reasoner_state import ReasonerState  # noqa:402

action_name_keys = {
    "catalyst_type": CatalystLabelChanger,
    "inclusion_criteria": IncludePropertyAdder,
    "exclusion_criteria": ExcludePropertyAdder,
    "relationship_to_candidate_list": RelationToCandidateListChanger,
}


class CoherentPolicy(BasePolicy):
    """A polocy like the Reasoner policy, but it promotes more coherent prompts."""

    def __init__(
        self,
        max_num_actions: int = 10,
    ):
        """Create the underlying ReasonerPolicy."""
        self.max_num_actions

    @classmethod
    @staticmethod
    def from_reasoner_policy(
        reasoner_policy: ReasonerPolicy, temperature: float = 0.6
    ) -> "CoherentPolicy":
        """Construct a coherent policy from a reasoner poliy."""
        p = CoherentPolicy()
        p.actions = reasoner_policy.actions.copy()
        p.init_weights()
        return p

    def get_actions(
        self, state: object
    ) -> tuple[list[Callable[object, object]], np.array]:
        """Return the actions along with their priors."""
        actions, priors = super().get_actions(state)
        # generate the trial states
        trial_states = []
        idx_trial_states = []  # mask for iompossible trial states
        for i, a in enumerate(actions):
            if priors[i] > 0:
                trial_states.append(a(state, trial=True))
                idx_trial_states.append(i)

        sim_scores = state.similarity(trial_states)

        full_sim_scores = np.zeros_like(priors)
        full_sim_scores[np.array(idx_trial_states)] = np.array(sim_scores)
        if state.reward is not None:
            reward_adjustment = full_sim_scores * (
                self.transform_reward(state.reward)
            ) + (1 - full_sim_scores) * (1 - self.transform_reward(state.reward))
        else:
            reward_adjustment = full_sim_scores

        state.info["priors"].update({"reward_adjusted_similarities": reward_adjustment})
        state.info["priors"].update(
            {"reward_adjustment_value": self.transform_reward(state.reward)}
        )

        new_priors = (
            softmax((reward_adjustment / self.temperature).astype(float)) * priors
        )
        new_priors = new_priors / np.sum(new_priors)  # re-normalize
        state.info["priors"].update({"values": new_priors})

        return actions, new_priors


def coherent_measure(
    states: list[ReasonerState], llm_function: callable = None
) -> float:
    """Measure the coherence of a given sequence of states."""
    prompts = []
    system_prompts = []
    answers = []
    for s in states:
        prompts.append(s.generation_prompt)
        system_prompts.append(s.generation_system_prompt)
        answers.append(s.answer)
    return -np.inf


if __name__ == "__main__":
    import pickle

    with open("data/example_trajectory.pkl", "rb") as f:
        states = pickle.load(f)

    coherent_measure(states)
