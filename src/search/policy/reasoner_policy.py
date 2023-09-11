"""Class for the reasoner policy."""
from collections.abc import Callable

import numpy as np


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

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        _add_property(new_state.include_list, self.property_name)
        if not trial:
            new_state.query()
        return new_state


class ExcludePropertyAdder:
    """Class to add property to a state."""

    def __init__(self, property_name):
        """Save the property name."""
        self.property_name = property_name

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        _add_property(new_state.exclude_list, self.property_name)
        if not trial:
            new_state.query()
        return new_state


_relationship_to_candidate_list_types = [
    "including elements that are different from",
    "including elements similar to",
    "introducing new elements to",
    "including elements from",
]


class RelationToCandidateListChanger:
    """Class to add property to a state."""

    def __init__(self, relationship_name):
        """Save the property name."""
        self.relationship_name = relationship_name

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        new_state.relation_to_candidate_list = self.relationship_name
        if not trial:
            new_state.query()
        return new_state


_catalyst_label_types = [
    "",
    "unary",
    "binary",
    "ternary",
]


class CatalystLabelChanger:
    """Class to add property to a state."""

    def __init__(self, catalyst_label_type):
        """Save the property name."""
        self.catalyst_label_type = catalyst_label_type

    def __call__(self, state, trial=False):
        """Add propery to the state."""
        new_state = state.return_next()
        if "oxide" in new_state.catalyst_label:
            new_state.catalyst_label = f"{self.catalyst_label_type} oxide cayalysts"
        else:
            new_state.catalyst_label = f"{self.catalyst_label_type} catalysts"
        if not trial:
            new_state.query()
        return new_state


def toggle_oxide(state, trial=False):
    """Toggle whether or not to target oxides."""
    new_state = state.return_next()
    if "oxide" in state.catalyst_label:
        new_state.catalyst_label = new_state.catalyst_label.replace(" oxide", "")
    else:  # Add in oxide to label
        new_state.catalyst_label = new_state.catalyst_label.replace(
            " catalysts", " oxide catalysts"
        )
    if not trial:
        new_state.query()
    return new_state


def _query_again(state, trial=False):
    new_state = state.return_next()
    if not trial:
        new_state.query()
    return new_state


class ReasonerPolicy:
    """Policy that modifes queries to an LLM."""

    def __init__(
        self,
        include_property_types: list[str] = None,
        exclude_property_types: list[str] = None,
        relationship_to_candidate_list_types: list[str] = None,
        catalyst_label_types: list[str] = None,
        try_oxides: bool = True,
    ):
        """Initialize the state and action pairs."""
        if include_property_types is None:
            include_property_types = _include_property_types
        if exclude_property_types is None:
            exclude_property_types = _exclude_property_types
        if relationship_to_candidate_list_types is None:
            relationship_to_candidate_list_types = _relationship_to_candidate_list_types
        if catalyst_label_types is None:
            catalyst_label_types = _catalyst_label_types

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
            self.actions.append(toggle_oxide)

        self.actions.append(_query_again)
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

    def check_relationship_to_candidate_list(self, state):
        """Ensure a relationship to candidate list exists if candidate list does."""
        if state.relation_to_candidate_list is None and state.candidates is not None:
            for i, a in enumerate(self.actions):
                if isinstance(a, RelationToCandidateListChanger):
                    self.weights[i] = 1
                else:
                    self.weights[i] = 0
        elif state.candidates is None:
            for i, a in enumerate(self.actions):
                if isinstance(a, RelationToCandidateListChanger):
                    self.weights[i] = 0
                else:
                    self.weights[i] = 1

    def get_actions(
        self, state: object
    ) -> tuple[list[Callable[object, object]], np.array]:
        """Return a actions and prior_logits for given state."""
        self.init_weights()
        self.check_repeated_properties(state)
        self.check_relationship_to_candidate_list(state)
        normalization = np.sum(self.weights) if np.sum(self.weights) != 0 else 1
        return (
            self.actions,
            self.weights / normalization,
        )

    @staticmethod
    def early_stopping(*args):
        """Whether to stop the search early. Always False for this policy."""
        return False
