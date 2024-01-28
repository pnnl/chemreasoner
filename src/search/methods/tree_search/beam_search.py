"""Search tree implemenation."""
import logging
from pathlib import Path
import pickle
import time

from random import shuffle

from typing import TypeVar

import numpy as np

SearchTree = TypeVar("SearchTree")

logging.getLogger().setLevel(logging.INFO)


class BeamSearchTree:
    """A class for running monte carlo tree search."""

    def __init__(
        self, data, policy, reward_fn, num_generate, num_keep, root_reward=True
    ):
        """Create a SearchTree from root node."""
        self.num_generate = num_generate
        self.num_keep = num_keep
        self.policy = policy
        self.reward_fn = reward_fn
        self.nodes = []
        self.nodes.append([data])
        self.parent_idx = [[-1]]
        if root_reward:
            self.node_rewards = [reward_fn([data])]
        else:
            self.node_rewards = [[0]]
        self.generated_nodes = [[]]
        self.generated_node_rewards = [[]]
        self.generated_parent_idx = [[]]

        self.start_time = None
        self.end_time = None
        # expand the root node

    def expand_node(self, nodes):
        """Expand out possible sub-nodes for a given list of nodes."""
        actions_priors = self.policy.get_actions(nodes)
        new_nodes = []
        parent_idx = []
        for i, node in enumerate(nodes):
            actions, these_priors = actions_priors[i]
            shuffle_idx = list(range(len(these_priors)))
            shuffle(shuffle_idx)
            these_priors = [these_priors[i] for i in shuffle_idx]
            actions = [actions[i] for i in shuffle_idx]

            action_idxs = np.argsort(these_priors)[-self.num_generate :]  # noqa: E203

            these_new_nodes = []
            for j in action_idxs:
                if these_priors[j] > 0:
                    a = actions[j]
                    these_new_nodes.append(a(node))
            new_nodes += these_new_nodes
            parent_idx += [i] * len(these_new_nodes)
        return new_nodes, parent_idx

    def simulation_policy(self):
        """Simulate a beam search step."""
        if self.start_time is None:
            self.start_timer()

        # expand final layer of nodes
        successor_nodes, parent_idx = self.expand_node(self.nodes[-1])
        try:
            # calculate their rewards
            successor_rewards = self.reward_fn(successor_nodes)
        except Exception as err:
            raise err
            logging.warning(
                "ERROR:Reward function call failed. Returning a penalty value."
            )
            print("ERROR:Reward function call failed. Returning a penalty value.")
            successor_rewards = [-10] * len(successor_nodes)

        # selected node index
        selected_node_idx = np.argsort(successor_rewards)[
            -self.num_keep :  # noqa: E203
        ]
        print(successor_rewards)
        generated_idx = np.argsort(successor_rewards)[: -self.num_keep]  # noqa: E203

        # Separate out the top-k rewards
        selected_nodes = [successor_nodes[i] for i in selected_node_idx]
        selected_rewards = [successor_rewards[i] for i in selected_node_idx]
        selected_parents = [parent_idx[i] for i in selected_node_idx]

        # Separate out the other nodes that were not chosen (generated_nodes)
        generated_nodes = [successor_nodes[i] for i in generated_idx]
        generated_node_rewards = [successor_rewards[i] for i in generated_idx]
        generated_parent_idx = [parent_idx[i] for i in generated_idx]

        # Save selected nodes
        self.nodes.append(selected_nodes)
        self.node_rewards.append(selected_rewards)
        self.parent_idx.append(selected_parents)

        # Save the generated_nodes
        self.generated_nodes.append(generated_nodes)
        self.generated_node_rewards.append(generated_node_rewards)
        self.generated_parent_idx.append(generated_parent_idx)

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
        """Turn beam search tree into dictionary for saving."""
        beam_search_data = dict()
        beam_search_data["nodes"] = []
        for list_nodes in self.nodes:
            beam_search_data["nodes"].append([vars(n) for n in list_nodes])
        beam_search_data["node_rewards"] = self.node_rewards
        beam_search_data["parent_idx"] = self.parent_idx

        beam_search_data["generated_nodes"] = []
        for list_nodes in self.generated_nodes:
            beam_search_data["generated_nodes"].append([vars(n) for n in list_nodes])
        beam_search_data["generated_node_rewards"] = self.generated_node_rewards
        beam_search_data["generated_parent_idx"] = self.generated_parent_idx

        beam_search_data["num_generate"] = self.num_generate
        beam_search_data["num_keep"] = self.num_keep

        beam_search_data["start_time"] = self.start_time
        beam_search_data["end_time"] = self.end_time

        return beam_search_data

    @classmethod
    @staticmethod
    def from_data(beam_search_data: dict, policy, reward_fn, node_constructor=None):
        """Create a beam search object from stored data."""
        new_tree = BeamSearchTree(
            None, policy, reward_fn, None, None, root_reward=False
        )

        for i, list_nodes in enumerate(beam_search_data["nodes"]):
            new_nodes = [node_constructor(n) for n in list_nodes]
            if i < len(new_tree.generated_nodes):
                new_tree.nodes[i] = new_nodes
            else:
                new_tree.nodes.append(new_nodes)

        new_tree.node_rewards = beam_search_data["node_rewards"]
        new_tree.parent_idx = beam_search_data["parent_idx"]

        for i, list_nodes in enumerate(beam_search_data["generated_nodes"]):
            new_nodes = [node_constructor(n) for n in list_nodes]
            if i < len(new_tree.generated_nodes):
                new_tree.generated_nodes[i] = new_nodes
            else:
                new_tree.generated_nodes.append(new_nodes)

        new_tree.generated_node_rewards = beam_search_data["generated_node_rewards"]
        new_tree.generated_parent_idx = beam_search_data["generated_parent_idx"]

        new_tree.num_generate = beam_search_data["num_generate"]
        new_tree.num_keep = beam_search_data["num_keep"]

        new_tree.start_time = beam_search_data["start_time"]
        new_tree.end_time = beam_search_data["end_time"]
        return new_tree

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

    def __len__(self):
        """Return the depth of self."""
        return len(self.nodes) - 1
