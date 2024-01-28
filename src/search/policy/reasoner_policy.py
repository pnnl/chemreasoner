"""Class for the reasoner policy."""
import sys
from collections.abc import Callable

import numpy as np

sys.path.append("sys")
from search.state.reasoner_state import ReasonerState  # noqa:402


_include_property_types = [
    "high activity",
    "high selectivity",
    "low cost",
    "novelty",
    "low toxicity",
    "high binding energy",
    "high selectivity",
    "high conversion",
    "high availability",
]

_exclude_property_types = [
    "low activity",
    "low selectivity",
    "low stability",
    "low binding energy",
    "high cost",
    "high toxicity",
    "low dispersion",
    "low porosity",
    "high scarcity",
    "low conversion",
]


def _add_property(property_list, property_name):
    property_list.append(property_name)


class IncludePropertyAdder:
    """Class to add property to a state."""

    def __init__(self, property_name):
        """Save the property name."""
        self.property_name = property_name
        self._message = (
            f"Include candidates with the good property {self.property_name}."
        )

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        _add_property(new_state.include_list, self.property_name)
        if not trial:
            pass
            # new_state.query()
        return new_state

    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message


class ExcludePropertyAdder:
    """Class to add property to a state."""

    def __init__(self, property_name):
        """Save the property name."""
        self.property_name = property_name
        self._message = (
            f"Exclude candidates with the bad property {self.property_name}."
        )

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        _add_property(new_state.exclude_list, self.property_name)
        if not trial:
            pass
            # new_state.query()
        return new_state

    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message


_relationship_to_candidate_list_types = [
    "include elements that are different from",
    "include elements similar to",
    "introduce new elements to",
    "include elements from",
]


class RelationToCandidateListChanger:
    """Class to add property to a state."""

    def __init__(self, relationship_name):
        """Save the property name."""
        self.relationship_name = relationship_name
        relationship_name_cap = relationship_name[0].upper() + relationship_name[1:]
        self._message = f"{relationship_name_cap} the predicted catalysts."

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        new_state.relation_to_candidate_list = self.relationship_name
        if not trial:
            pass
            # new_state.query()
        return new_state

    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message


_catalyst_label_types = [
    "metallic catalysts",
    "monometallic catalysts",
    "bimetallic catalysts",
    "trimetallic catalysts",
]


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


class ToggleOxide:
    @staticmethod
    def __call__(state, trial=False):
        """Toggle whether or not to target oxides."""
        new_state = state.return_next()
        if "oxide" in state.catalyst_label:
            new_state.catalyst_label = new_state.catalyst_label.replace("oxide ", "")
        else:  # Add in oxide to label
            new_state.catalyst_label = new_state.catalyst_label.replace(
                "catalysts", "oxide catalysts"
            )
        if not trial:
            pass
            # new_state.query()
        return new_state

    @staticmethod
    def message(state):
        """Return the message for this action. Depends on the state."""
        if "oxide" in state.catalyst_label:
            return "Search for non-oxide catalysts, instead."
        else:  # Add in oxide to label
            return "Search for oxide catalysts, instead."


class QueryAgain:
    _message = "Run the same query again."

    @staticmethod
    def __call__(state, trial=False):
        new_state = state.return_next()
        if not trial:
            pass
            # new_state.query()
        return new_state

    def message(self, state):
        """Return the message for this action. State does nothing."""
        return self._message


class ReasonerPolicy:
    """Policy that modifes queries to an LLM."""

    def __init__(
        self,
        include_property_types: list[str] = _include_property_types,
        exclude_property_types: list[str] = _exclude_property_types,
        relationship_to_candidate_list_types: list[
            str
        ] = _relationship_to_candidate_list_types,
        catalyst_label_types: list[str] = _catalyst_label_types,
        try_oxides: bool = False,
    ):
        """Initialize the state and action pairs."""
        self.actions = []
        for prop in include_property_types:
            self.actions.append(IncludePropertyAdder(prop))
        for prop in exclude_property_types:
            self.actions.append(ExcludePropertyAdder(prop))
        for rel in relationship_to_candidate_list_types:
            self.actions.append(RelationToCandidateListChanger(rel))
        label_types = catalyst_label_types
        for label in label_types:
            self.actions.append(CatalystLabelChanger(label))

        if try_oxides:
            self.actions.append(ToggleOxide)

        self.actions.append(QueryAgain())
        self.init_weights()

    def init_weights(self):
        """Re initialzie the weights."""
        self.weights = np.ones_like(self.actions)

    def check_repeated_properties(self, state):
        """To prevent duplicate properties, set weights to zero."""
        for i, a in enumerate(self.actions):
            if isinstance(a, IncludePropertyAdder) or isinstance(
                a, ExcludePropertyAdder
            ):
                if a.property_name in state.include_list + state.exclude_list:
                    self.weights[i] = 0

    def check_repeated_catalyst_type(self, state):
        """Prevent changing to the same catalyst type."""
        for i, a in enumerate(self.actions):
            if isinstance(a, CatalystLabelChanger):
                if a.catalyst_label_type == state.catalyst_label:
                    self.weights[i] = 0

    def check_relationship_to_candidate_list(self, state):
        """Ensure a relationship to candidate list exists if candidate list does."""
        if state.relation_to_candidate_list is None and len(state.candidates) != 0:
            for i, a in enumerate(self.actions):
                if isinstance(a, RelationToCandidateListChanger):
                    self.weights[i] = 1
                else:
                    self.weights[i] = 0
        elif len(state.candidates) == 0:
            for i, a in enumerate(self.actions):
                if isinstance(a, RelationToCandidateListChanger):
                    self.weights[i] = 0
                else:
                    self.weights[i] = 1

    def get_actions(
        self, states: list[ReasonerState]
    ) -> tuple[list[Callable], list[np.array]]:
        """Return a actions and prior_logits for given state."""
        action_priors = []
        for state in states:
            self.init_weights()
            self.check_repeated_properties(state)
            self.check_relationship_to_candidate_list(state)
            normalization = np.sum(self.weights) if np.sum(self.weights) != 0 else 1
            action_priors.append((self.actions, self.weights / normalization))
        return action_priors

    @staticmethod
    def early_stopping(*args):
        """Whether to stop the search early. Always False for this policy."""
        return False
