import logging
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from networkx.drawing.nx_pydot import graphviz_layout

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator
from llm.azure_open_ai_interface import AzureOpenaiInterface
from search.reward.microstructure_reward import MicrostructureRewardFunction
from structure_creation.digital_twin import CatalystDigitalTwin
from structure_creation.microstructure_planner import OCPMicrostructurePlanner


logging.getLogger().setLevel(logging.INFO)


class MicrostructureTree:
    nodes = {}

    def __init__(self, root_node: CatalystDigitalTwin):
        """Initialize self with given root_node."""
        self.nodes[root_node._id] = root_node
        self.root_id = root_node._id

    def set_children(self, parent_id, children_nodes: list[CatalystDigitalTwin]):
        """Add the children to the given parent."""
        parent_node = self.nodes[parent_id]
        for child in children_nodes:
            self.nodes[child._id] = child
            parent_node.children_ids.append(child._id)

    def get_children(self, node_id):
        """Get the children for the given node_id."""
        return self.nodes[node_id].children_ids

    def get_node_value(self, node_id, storage_dict: dict = None):
        """Get the downstream rewards of node_id."""
        r, n = self.get_downstream_rewards_and_leaf_nodes(
            node_id=node_id, storage_dict=storage_dict
        )
        if storage_dict is not None:
            for k, v in storage_dict.items():
                storage_dict[k] = v[0] / v[1]
        return r / n

    def get_downstream_rewards_and_leaf_nodes(
        self, node_id, reward_agg_func=sum, storage_dict: dict = None
    ):
        """Get the downstream rewards and leaf nodes of node_id returning as tuple."""
        node = self.nodes[node_id]

        if node.get_reward() is not None:
            return_value = (node.get_reward(), 1)

        elif node.get_reward() is None and len(self.get_children(node_id)) > 0:

            rewards, children = zip(
                *[
                    self.get_downstream_rewards_and_leaf_nodes(
                        _id, reward_agg_func=reward_agg_func, storage_dict=storage_dict
                    )
                    for _id in node.children_ids
                ]
            )
            return_value = (reward_agg_func(rewards), sum(children))
        else:
            logging.warning(f"No simulations have been run for leaf node {node_id}.")
            return_value = None

        if storage_dict is not None:
            storage_dict[node_id] = return_value
        print(return_value)
        return return_value

    def get_downstream_rewards(
        self, node_id, reward_agg_func=sum, storage_dict: dict = None
    ):
        """Get the downstream rewards of node_id."""
        node = self.nodes[node_id]
        if node.get_reward() is not None:
            return_value = node.get_reward()
        elif node.get_reward() is None and len(self.get_children(node_id)) > 0:
            rewards = [
                self.get_downstream_rewards(
                    _id, agg_func=reward_agg_func, storage_dict=storage_dict
                )
                for _id in node.children_ids
            ]
            return_value = reward_agg_func(rewards)
        else:
            logging.warning(f"No simulations have been run for leaf node {node_id}.")
            return_value = None

        if storage_dict is not None:
            storage_dict[node_id] = return_value
        return return_value

    def get_downstream_leaf_nodes(self, node_id, storage_dict: dict = None):
        """Get the downstream rewards of node_id."""
        node = self.nodes[node_id]
        if node.get_reward() is not None:
            return_value = 1
        elif node.get_reward() is None and len(self.get_children(node_id)) > 0:
            children = [
                self.get_downstream_leaf_nodes(_id, storage_dict=storage_dict)
                for _id in node.children_ids
            ]
            return_value = sum(children)
        else:
            logging.warning(f"No simulations have been run for leaf node {node_id}.")
            return_value = None

        if storage_dict is not None:
            storage_dict[node_id] = return_value
        return return_value

    def get_leaf_nodes(self):
        """Return the list of leaf nodes for self."""
        leaf_nodes = []
        self._get_leaf_nodes_recurse(self.root_id, leaf_nodes_list=leaf_nodes)
        return leaf_nodes

    def _get_leaf_nodes_recurse(self, current_node_id, leaf_nodes_list: list):
        """Get the list of leaf nodes below the given node."""
        children = self.get_children(current_node_id)
        if len(children) > 0:
            for c_id in children:
                self._get_leaf_nodes_recurse(c_id, leaf_nodes_list=leaf_nodes_list)
        else:
            leaf_nodes_list.append(current_node_id)

    def to_nx(self):
        """Return a networkx graph representing self."""
        edges = [
            (node_id, child_id)
            for node_id in self.nodes.keys()
            for child_id in self.get_children(node_id=node_id)
        ]
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G


dt = CatalystDigitalTwin()
syms = ["Cu", "Zn"]
dt.computational_params["symbols"] = syms
dt.computational_objects["symbols"] = syms

tree = MicrostructureTree(root_node=dt)


def microstructure_search(
    tree: MicrostructureTree,
    microstructure_planner: OCPMicrostructurePlanner,
):
    """Run the search logic for the given tree."""
    root_id = tree.root_id
    children = tree.get_children(root_id)
    if len(children) == 0:
        nodes = [tree.nodes[root_id]]
        bulks_idxs = [[0, 1, 2]] * len(nodes)  # ms_planner.run_bulk_prompt(nodes)
        for i in range(len(nodes)):
            parent_node = nodes[i]

            these_bulks = bulks_idxs[i]
            available_bulks = parent_node.get_bulks()
            selected_bulks = [available_bulks[j] for j in these_bulks]
            # Generate child nodes and put them in the tree
            tree.set_children(parent_node._id, parent_node.set_bulk(selected_bulks))

    # set the millers
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]

    millers_choices = [[(1, 1, 1), (1, 1, 0), (1, 0, 0)]] * len(
        nodes
    )  # ms_planner.run_millers_prompt(nodes)
    print(millers_choices)
    for i in range(len(nodes)):
        parent_node = nodes[i]
        print(parent_node._id)
        print(len(tree.get_children(parent_node._id)))
        these_millers = millers_choices[i]
        print(i)
        print(these_millers)
        # Generate child nodes and put them in the tree
        tree.set_children(parent_node._id, parent_node.set_millers(these_millers))
        print(len(parent_node.children_ids))

    # set the surface
    print(len(nodes))
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    print(len(nodes))
    for n in nodes:
        print(list(n.computational_objects.keys()))
    surface_choices = [
        n.get_surfaces()[:1] for n in nodes
    ]  # ms_planner.run_millers_prompt(nodes)
    print(surface_choices)
    for i in range(len(nodes)):
        parent_node = nodes[i]

        these_surfaces = surface_choices[i]
        # Generate child nodes and put them in the tree
        tree.set_children(parent_node._id, parent_node.set_surfaces(these_surfaces))

    # get the nodes
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    site_placement_choices = [
        n.get_site_placements()[:10] for n in nodes
    ]  # ms_planner.run_site_placement_prompt(nodes)
    print(site_placement_choices)
    for i in range(len(nodes)):
        parent_node = nodes[i]

        these_site_placements = site_placement_choices[i]
        # Generate child nodes and put them in the tree
        tree.set_children(
            parent_node._id, parent_node.set_site_placements(these_site_placements)
        )

    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    return nodes


def visualize_tree(tree: MicrostructureTree):
    """Visualize the given microstructure tree.

    Uses hue for node values and computational params for labels."""
    node_values = {}
    tree.get_node_value(tree.root_id, storage_dict=node_values)
    print(node_values)
    node_labels = {
        k: (
            node.computational_params[node.status]
            if not isinstance(node.computational_params[node.status], tuple)
            else simplify_float_values(node.computational_params[node.status])
        )
        for k, node in tree.nodes.items()
    }
    T = tree.to_nx()

    node_color = [node_values[n] for n in T.nodes()]
    vmin = min(node_color)
    vmax = max(node_color)
    pos = graphviz_layout(T, prog="dot")
    text = nx.draw_networkx_labels(T, pos=pos, labels=node_labels, font_size=8)

    for _, t in text.items():
        t.set_rotation(45)

    nx.draw(T, pos=pos, labels=node_labels, node_color=node_color, with_labels=False)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    ax = plt.gca()
    plt.colorbar(
        sm,
        cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),
    )


def simplify_float_values(tuple_data: tuple):
    """Simplify the floating point values in the given tuple."""
    return tuple(
        "{:.2f}".format(element) if isinstance(element, float) else element
        for element in tuple_data
    )


if __name__ == "__main__":

    class TestState:
        root_prompt = "Propose a catalyst for the conversion of CO to methanol."

    pathways = [
        ["*CO", "*COH", "CHOH", "*CH2OH", "*OHCH3"],
        ["*CO", "*CHO", "CHOH", "*CH2OH", "*OHCH3"],
    ]
    calc = OCAdsorptionCalculator(
        **{
            "model": "gemnet-oc-22",
            "traj_dir": Path("test_trajs"),
            "batch_size": 45,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.03,
            "steps": 3,
        }
    )
    reward_func = MicrostructureRewardFunction(
        pathways, calc, num_augmentations_per_site=1
    )

    state = TestState()

    llm_function = AzureOpenaiInterface(dotenv_path=".env", model="gpt-4")
    ms_planner = OCPMicrostructurePlanner(llm_function=llm_function)
    ms_planner.set_state(state)
    nodes = microstructure_search(tree, ms_planner)
    rewards = reward_func(nodes)
    for r, n in zip(rewards, nodes):
        n.set_reward(r)

    print(rewards)

    visualize_tree(tree=tree)
    plt.title("**Placeholder values for rewards and catalyst values**")

    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig("test_tree.png", dpi=300)
    plt.show()
