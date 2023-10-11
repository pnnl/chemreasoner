"""Class for the coherence policy."""
import sys

from collections.abc import Callable

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

sys.path.append("src")
from search.policy.reasoner_policy import ReasonerPolicy  # noqa:402


class LLMDrivenReasonerPolicy(ReasonerPolicy):
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
    ) -> "LLMDrivenReasonerPolicy":
        """Construct a coherent policy from a reasoner poliy."""
        p = LLMDrivenReasonerPolicy()
        p.actions = reasoner_policy.actions.copy()
        p.init_weights()
        return p

    def get_actions(
        self, states: object
    ) -> tuple[list[Callable[object, object]], np.array]:
        """Return the actions along with their priors."""
        actions, priors = super().get_actions(
            states
        )  # get super class actions and priors
        new_priors = []
        for i, state in enumerate(states):

            prev_answer = state.answer

            prior_prompt = f"Consider the previous answer:\n{prev_answer}.\n\n"
            prior_prompt += f"The reward for this prompt was {state.reward}.\n\n"
            prior_prompt += (
                "Your task is to rate the following actions to produce a "
                "new prompt that an llm can use to recommend better catalysts.\n\n"
            )

            actions_statement = "The actions are:\n"
            for i, a in enumerate(actions):
                if priors[i] != 0:
                    actions_statement += (
                        f"{i}) {a.message(s)}\n"  # punctuation is in a.message
                    )

            prior_prompt += actions_statement
            prior_prompt += (
                "\nReturn your answer as a python dictionary mapping each "
                "action to a score from 0 to 10 (10 is the best)."
            )

            print(prior_prompt)
            new_priors.append(priors[i])

        return actions, new_priors


class TestState:
    answer = """To generate a list of top-5 monometallic catalysts for the adsorption of *CH2CH2OH, we need to consider catalysts that can effectively interact with the adsorbate and promote its adsorption. Here are the top-5 catalysts along with their scientific explanations:

1. Platinum (Pt):
Platinum is a highly effective catalyst for the adsorption of *CH2CH2OH due to its ability to form strong bonds with oxygen-containing species. The Pt surface can adsorb *CH2CH2OH through the dissociation of the C-O bond, leading to the formation of *CH2CH2 and *OH species. This dissociation step is facilitated by the high reactivity of Pt towards oxygen-containing compounds.

2. Palladium (Pd):
Palladium is another excellent catalyst for the adsorption of *CH2CH2OH. Similar to Pt, Pd can form strong bonds with oxygen-containing species. The Pd surface can adsorb *CH2CH2OH by breaking the C-O bond, resulting in the formation of *CH2CH2 and *OH species. Pd is known for its high catalytic activity and selectivity towards oxygen-containing compounds.

3. Silver (Ag):
Silver is a promising catalyst for the adsorption of *CH2CH2OH due to its ability to interact with oxygen-containing species. The Ag surface can adsorb *CH2CH2OH by breaking the C-O bond, leading to the formation of *CH2CH2 and *OH species. Ag exhibits good catalytic activity and stability, making it a suitable catalyst for this reaction.

4. Rhodium (Rh):
Rhodium is a highly effective catalyst for the adsorption of *CH2CH2OH due to its strong interaction with oxygen-containing species. The Rh surface can adsorb *CH2CH2OH by breaking the C-O bond, resulting in the formation of *CH2CH2 and *OH species. Rh is known for its excellent catalytic properties and can promote the adsorption of oxygen-containing compounds.

5. Ruthenium (Ru):
Ruthenium is a versatile catalyst for the adsorption of *CH2CH2OH due to its ability to interact with oxygen-containing species. The Ru surface can adsorb *CH2CH2OH by breaking the C-O bond, leading to the formation of *CH2CH2 and *OH species. Ru exhibits high catalytic activity and stability, making it a suitable catalyst for this reaction.

Now, let's return the python list named final_answer containing the top-5 catalysts:

final_answer = ['Platinum (Pt)', 'Palladium (Pd)', 'Silver (Ag)', 'Rhodium (Rh)', 'Ruthenium (Ru)']"""
    reward = 30
    candidates = [
        "Platinum (Pt)",
        "Palladium (Pd)",
        "Silver (Ag)",
        "Rhodium (Rh)",
        "Ruthenium (Ru)",
    ]

    def __init__(
        self,
        catalyst_label: str = "catalysts",
        relation_to_candidate_list: str = None,
        include_list: list[str] = [],
        exclude_list: list[str] = [],
    ):

        self.catalyst_label = catalyst_label
        self.relation_to_candidate_list = "similar to"
        self.include_list = ["low cost", "high activity"]
        self.exclude_list = ["high cost", "low conversion"]


if __name__ == "__main__":
    from llm.automate_prompts import get_initial_state_oc

    s = TestState()
    p = LLMDrivenReasonerPolicy(0.4)
    print(p.get_actions(s))
