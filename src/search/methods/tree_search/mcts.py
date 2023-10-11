"""Search tree implemenation."""
from pathlib import Path
import pickle
import time
import warnings

from typing import TypeVar, Tuple

import numpy as np

SearchTree = TypeVar("SearchTree")


class MonteCarloTree:
    """A class for running monte carlo tree search."""

    def __init__(self, data, policy, reward_fn, tradeoff, discount_factor):
        """Create a SearchTree from root node."""
        self.nodes = []
        self.node_rewards = np.array([])
        self.children_idx = np.array([])
        self.children_priors = np.array([])
        self.children_rewards = np.array([])
        self.children_visits = np.array([])

        self.tradeoff = tradeoff
        self.discount_factor = discount_factor
        self.policy = policy
        self.reward_fn = reward_fn
        self.nodes = []
        self.nodes.append(data)
        # self.parents = np.append(self.parents, -1)
        self.node_rewards = np.append(self.node_rewards, 0)
        # expand the root node
        self.expand_root_node()

        self.start_time = None
        self.end_time = None

    def expand_root_node(self):
        """Expand out the children rewards and values for the root node."""
        _, priors = self.policy.get_actions(self.nodes[0])
        num_actions = len(priors)
        self.children_idx = np.zeros((1, num_actions), dtype=int) - 1
        self.children_rewards = np.zeros((1, num_actions))
        self.children_values = np.zeros((1, num_actions))
        self.children_visits = np.zeros((1, num_actions))
        self.children_priors = priors[np.newaxis, :]

    def add_node(self, current_node, priors):
        """Expand out the children rewards and values for the root node."""
        self.nodes.append(current_node)
        num_actions = len(self.policy.get_actions(self.nodes[0])[0])
        self.children_idx = np.concatenate(
            (self.children_idx, np.zeros((1, num_actions), dtype=int) - 1), axis=0
        )
        self.children_rewards = np.concatenate(
            (self.children_rewards, np.zeros((1, num_actions))), axis=0
        )
        self.children_values = np.concatenate(
            (self.children_values, np.zeros((1, num_actions))), axis=0
        )
        self.children_visits = np.concatenate(
            (self.children_visits, np.zeros((1, num_actions))), axis=0
        )
        self.children_priors = np.concatenate(
            (self.children_priors, priors[np.newaxis, :]), axis=0
        )

    def backup(
        self,
        history: list[Tuple[int, int]],
        reward: float,
        value: int = 0,
    ):
        """Backup reward through the given history."""
        current_discount = 1
        while len(history) > 0:
            current_idx, action_idx = history.pop()

            self.children_rewards[current_idx, action_idx] += current_discount * reward
            self.children_values[current_idx, action_idx] += current_discount * value
            self.children_visits[current_idx, action_idx] += 1

            current_discount *= self.discount_factor

        return

    def simulation_policy(self, starting_idx: int = 0):
        """Simulate the tree from the root node."""
        if self.start_time is None:
            self.start_timer()
        history_stack = []
        current_idx = 0
        current_node = self.nodes[0]

        continue_simulating = True
        while continue_simulating:
            children_visits = self.children_visits[current_idx]
            children_rewards = self.children_rewards[current_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Q = np.nan_to_num(
                    children_rewards / children_visits, posinf=0
                )  # Action value

            priors = self.children_priors[current_idx]
            root_total_visits = np.sqrt(np.sum(children_visits))
            if root_total_visits == 0:
                u = priors
            else:
                u = priors * root_total_visits / (1 + children_visits)

            action_value = np.array(Q + self.tradeoff * u, dtype=float)
            action_idx = np.random.choice(
                np.flatnonzero(np.isclose(action_value, action_value.max()))
            )  # Take max action value
            history_stack.append((current_idx, action_idx))
            if self.children_idx[current_idx, action_idx] != -1:
                # Choose old node
                current_idx = self.children_idx[current_idx][action_idx]
                current_node = self.nodes[current_idx]
                # TODO: Re-sample non-deterministic action?
            else:
                continue_simulating = False
                actions, _ = self.policy.get_actions(current_node)
                # Generate the new leaf node
                current_node = actions[action_idx](current_node)
                _, priors = self.policy.get_actions(
                    current_node
                )  # calculate new priors
                priors = priors[0]
                if sum(priors.flatten()) != 0:
                    self.add_node(current_node, priors)
                    self.children_idx[current_idx, action_idx] = len(self.nodes) - 1
                    current_idx = len(self.nodes) - 1

                    reward = self.reward_fn([current_node])[0]
                    self.node_rewards = np.append(self.node_rewards, reward)
                else:  # Further progressing this state is not possible.
                    history_stack.pop()
                    reward = 0

        self.backup(history_stack, reward)

    def get_best_state(self, reward=False):
        """Return the state with the lowest reward."""
        idx = np.argmax(self.node_rewards)
        if reward:
            return (self.nodes[idx], self.node_rewards[idx])
        return self.nodes[idx]

    def get_branching_factor(self):
        """Return the branching factor of the search tree."""
        return self.children_idx.shape[0] / np.sum(
            np.any(self.children_idx != -1, axis=1)
        )

    def start_timer(self):
        """Save the time to the start time."""
        self.start_time = time.time()

    def end_timer(self):
        """Save a number to the end timer."""
        self.end_time = time.time()

    def get_time(self):
        """Save a number to the end timer."""
        return self.end_time - self.start_time

    def reset_timer(self):
        """Reset the time values to None."""
        self.start_time = None
        self.end_time = None

    def get_processed_data(self) -> dict:
        """Cast a search tree into a format that can be read."""
        mcts_data = dict()
        mcts_data["discount"] = self.discount_factor
        mcts_data["tradeoff"] = self.tradeoff
        mcts_data["nodes"] = [vars(n) for n in self.nodes]
        mcts_data["children_idx"] = self.children_idx
        mcts_data["children_priors"] = self.children_priors
        mcts_data["children_rewards"] = self.children_rewards
        mcts_data["children_visits"] = self.children_visits
        mcts_data["node_rewards"] = self.node_rewards

        mcts_data["start_time"] = self.start_time
        mcts_data["end_time"] = self.end_time

        return mcts_data

    def pickle(self, fname: Path):
        """Save beam search to pickle file."""
        pickle_data = self.get_processed_data()
        with open(fname, "wb") as f:
            pickle.dump(pickle_data, f)

    def step_return(self):
        """Take a step and return the tree data."""
        self.simulation_policy()
        self.end_timer()
        return self.get_processed_data()

    def step_save(self, fname):
        """Take a simulation step and save the resulting tree state with end_time."""
        self.simulation_policy()
        self.end_timer()
        self.pickle(fname)
