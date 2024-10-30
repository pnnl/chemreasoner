"""Code to """

import argparse
import json
import logging
import math
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ase.data import chemical_symbols
from ase.io import write

from networkx.drawing.nx_pydot import graphviz_layout

sys.path.append("src")
from nnp.oc import OCAdsorptionCalculator
from nnp.uncertainty_prediction import UncertaintyCalculator
from llm.azure_open_ai_interface import AzureOpenaiInterface
from search.reward.microstructure_reward import (
    MicrostructureRewardFunction,
    MicrostructureUncertaintyFunction,
)
from structure_creation.digital_twin import CatalystDigitalTwin
from structure_creation.microstructure_planner import (
    OCPMicrostructurePlanner,
    describe_site_placement,
)


logging.getLogger().setLevel(logging.INFO)


class MicrostructureTree:

    def __init__(self, root_node: CatalystDigitalTwin):
        """Initialize self with given root_node."""
        self.nodes = {}
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
        self,
        node_id,
        reward_agg_func=sum,
        storage_dict: dict = None,
        uncertainty: bool = False,
    ):
        """Get the downstream rewards and leaf nodes of node_id returning as tuple."""
        node = self.nodes[node_id]

        if node.get_reward() is not None:
            if not uncertainty:
                return_value = (node.get_reward(), 1)
            else:
                return_value = (node.get_reward(), 1, node.get_uncertainty())

        elif node.get_reward() is None and len(self.get_children(node_id)) > 0:

            if not uncertainty:
                rewards, children = zip(
                    *[
                        self.get_downstream_rewards_and_leaf_nodes(
                            _id,
                            reward_agg_func=reward_agg_func,
                            storage_dict=storage_dict,
                            uncertainty=uncertainty,
                        )
                        for _id in node.children_ids
                    ]
                )
                return_value = (reward_agg_func(rewards), sum(children))
            else:
                rewards, children, uq = zip(
                    *[
                        self.get_downstream_rewards_and_leaf_nodes(
                            _id,
                            reward_agg_func=reward_agg_func,
                            storage_dict=storage_dict,
                            uncertainty=uncertainty,
                        )
                        for _id in node.children_ids
                    ]
                )
                return_value = (
                    reward_agg_func(rewards),
                    sum(children),
                    np.sqrt(np.sum(np.array(uq) ** 2)),
                )
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

    def get_leaf_nodes(self):  # TODO: Add root node parameter for downstream leaves
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

    def store_data(
        self, metadata: bool = False
    ) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
        """Save the data stored in self."""
        metadata_dict = {} if metadata else None
        node_data = []
        edge_data = []
        for n_id, n in self.nodes.items():
            if metadata:
                data, info = n.return_row(metadata=metadata)
                metadata_dict.update({n._id: info})
            else:
                data = n.return_row()
            node_data.append(data)
            edge_data += [[n_id, c] for c in n.children_ids]
        if metadata:
            return pd.DataFrame(node_data), edge_data, metadata_dict
        else:
            return pd.DataFrame(node_data), edge_data

    @classmethod
    def from_data(
        cls,
        node_data: pd.DataFrame,
        edge_data: list[tuple[str, str]],
        node_constructor=CatalystDigitalTwin.from_row,
    ) -> "MicrostructureTree":
        """Return a microstructure tree from the given data."""
        edge_data = edge_data.copy()
        node_dict = {}
        edge_dict = {}
        root_node = False
        root_id = None
        for i, row in node_data.iterrows():
            node = node_constructor(row)
            node_dict[node._id] = node

            j = 0
            edge_dict[node._id] = []
            root_node = True
            while j < len(edge_data):
                e = edge_data[j]
                if (
                    root_id is None and e[1] == node._id
                ):  # Check if this isn't the root node
                    root_node = False

                # print(e[0], node._id)
                if e[0] == node._id:  # Check if this node is a parent to this edge
                    edge_dict[node._id].append(edge_data.pop(j)[1])
                    # edge_data.pop(j)
                else:
                    j += 1

            if root_id is None and root_node:
                print(f"{node._id} found root node *********")
                root_id = node._id
        tree = cls(root_node=node_dict[root_id])

        # Add the children. Must be in order from root node so parent node is
        # already in the tree.
        def _recursive_add_children(tree, node_id):
            children = [node_dict[c_id] for c_id in edge_dict[node_id]]
            tree.set_children(node_id, children)
            for c_id in edge_dict[node_id]:
                _recursive_add_children(tree, c_id)

        _recursive_add_children(tree, root_id)
        return tree


def microstructure_search(
    tree: MicrostructureTree,
    microstructure_planner: OCPMicrostructurePlanner,
):
    """Run the search logic for the given tree."""
    root_id = tree.root_id
    if (
        len(tree.nodes[root_id].get_bulks())
        > microstructure_planner.num_choices["bulk"]
    ):
        nodes = [tree.nodes[root_id]]
        # bulks_idxs = [[0, 1, 2]] * len(nodes)
        bulks_idxs = ms_planner.run_bulk_prompt(nodes)
        for i in range(len(nodes)):
            parent_node = nodes[i]

            these_bulks = bulks_idxs[i]
            available_bulks = parent_node.get_bulks()
            selected_bulks = [available_bulks[j] for j in these_bulks]
            # Generate child nodes and put them in the tree
    else:
        parent_node = tree.nodes[root_id]
        nodes = [parent_node]
        selected_bulks = parent_node.get_bulks()
    tree.set_children(parent_node._id, parent_node.set_bulk(selected_bulks))

    # set the millers
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]

    # millers_choices = [[(1, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]] * len(
    #     nodes
    # )
    millers_choices = ms_planner.run_millers_prompt(nodes)
    print(millers_choices)
    for i in range(len(nodes)):
        parent_node = nodes[i]
        these_millers = millers_choices[i]
        # Generate child nodes and put them in the tree
        tree.set_children(parent_node._id, parent_node.set_millers(these_millers))

    # set the surface (Use the first surface that shows up in each case)
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    surface_choices = [n.get_surfaces()[:1] for n in nodes]
    for i in range(len(nodes)):
        parent_node = nodes[i]

        these_surfaces = surface_choices[i]
        # Generate child nodes and put them in the tree
        tree.set_children(parent_node._id, parent_node.set_surfaces(these_surfaces))

    # get the nodes
    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    # site_placement_choices = [n.get_site_placements()[:8] for n in nodes]
    site_placement_choices = ms_planner.run_site_placement_prompt(nodes)
    for i in range(len(nodes)):
        parent_node = nodes[i]

        these_site_placements = site_placement_choices[i]
        # Generate child nodes and put them in the tree
        tree.set_children(
            parent_node._id, parent_node.set_site_placements(these_site_placements)
        )

    nodes = [tree.nodes[child] for n in nodes for child in tree.get_children(n._id)]
    return nodes


def microstructure_finetune_selection(
    tree: MicrostructureTree,
    # microstructure_planner: OCPMicrostructurePlanner,
    top_k: int,
    percentile_reward=0.75,
):
    """Run the search logic for the given tree."""
    leaf_nodes = [tree.nodes[n] for n in tree.get_leaf_nodes()]
    percentile_r = np.percentile(
        [n.get_reward() for n in leaf_nodes], 1 - percentile_reward
    )
    leaf_nodes = [n for n in leaf_nodes if n.get_reward() > percentile_r]

    best_nodes = sorted(leaf_nodes, key=lambda n: n.get_reward() * n.get_uncertainty())[
        -top_k:
    ]
    return [n._id for n in best_nodes]


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


def extract_dft_candidates(
    dataframe: pd.DataFrame,
    num_samples: int,
    columns=["symbols", "bulk_composition"],
):
    """Extract dft candidates for a given reward dataframe, stratifying on the given columns."""
    sampling_priors, sample_names = _recursive_get_sampling_priors(
        dataframe=dataframe,
        columns=columns,
        sampling_prior=1.0,
        total_num_samples=num_samples,
    )
    return sampling_priors, sample_names


def _sample_dataframe(dataframe: pd.DataFrame, num_samples: int):
    """Sample the energies to run DFT with."""
    uncertainty_columns = [col for col in dataframe.columns if "uncertainty" in col]
    # gather the uncertainty values
    uncertainty_values = []
    for i, row in dataframe.iterrows():
        for col in uncertainty_columns:
            if "e_slab" not in uncertainty_values:
                new_col = col.replace("uncertainty_", "")
                uncertainty_values.append((f"{row['id']}_{new_col}", row[col]))
            else:
                # Check to make sure the slab uncertainty is not repeated
                unique = True
                for val in uncertainty_values:
                    if "e_slab" in val[0] and row[col] == val[1]:
                        unique = False
                if unique:
                    uncertainty_values.append((f"{row['id']}_{new_col}", row[col]))
    samples = sorted(uncertainty_values, key=lambda x: x[1])[-num_samples:]
    return samples


def _recursive_get_sampling_priors(
    dataframe: pd.DataFrame,
    columns: list,
    sampling_prior: float,
    total_num_samples: int,
):
    """Recursively get samples from the given dataframe."""
    column_name = columns[0]
    column_values = dataframe[column_name].unique()
    sampling_priors = {}
    sampling_rewards = {}
    for val in column_values:
        sampling_rewards[val] = np.nanmean(
            dataframe[dataframe[column_name] == val]["reward"]
        )

    sorted_column_rewards = sorted(
        [v if not np.isnan(v) else -np.inf for k, v in sampling_rewards.items()]
    )
    sampling_priors = {
        column_values[i]: pow(2, -(i + 1)) for i, _ in enumerate(sorted_column_rewards)
    }
    sampling_priors = _normalize_priors(sampling_priors)

    if len(columns) > 1:
        return_data = {}
        samples = []
        for k, p in sampling_priors.items():
            return_data[k], samples_ = _recursive_get_sampling_priors(
                dataframe=dataframe[dataframe[column_name].to_numpy() == k],
                columns=columns[1:],
                sampling_prior=p * sampling_prior,
                total_num_samples=total_num_samples,
            )
            samples += samples_
        return return_data, samples
    else:
        return_data = {}
        samples = []
        for k, p in sampling_priors.items():
            return_data[k] = p * sampling_prior
            samples += _sample_dataframe(
                dataframe[dataframe[column_name].to_numpy() == k],
                math.ceil(total_num_samples * p),
            )
        return return_data, samples


def _normalize_priors(priors: dict):
    """Normalize the prior probabilities given in the priors."""
    N = sum(list(priors.values()))
    return {k: v / N for k, v in priors.items()}


def simplify_float_values(tuple_data: tuple):
    """Simplify the floating point values in the given tuple."""
    return tuple(
        "{:.2f}".format(element) if isinstance(element, float) else element
        for element in tuple_data
    )


class _PlaceholderAtomisticCalc:
    """A placehodler class for an atomistic calculator."""

    def __init__(self, traj_dir: Path):
        """Initialize self with the given traj_dir"""
        self.data = traj_dir


class MicrostructureRewardAnalyzer:
    """Class to anaylze microstructure reward values from a tree."""

    def __init__(
        self, tree: MicrostructureTree, pathways: list[list[str]], traj_dir: Path
    ):
        """Microstructure tree search."""
        self.tree = tree
        self.pathways = pathways
        self.traj_dir = traj_dir

        self.calc = _PlaceholderAtomisticCalc(traj_dir=traj_dir)

    def colllect_calculated_values(self):
        """Collect the values of adsorption energies collected for each node."""
        nodes = tree.get_leaf_nodes()


def get_reward_data(
    tree: MicrostructureTree,
    reward_func: MicrostructureRewardFunction,
    uq_func: MicrostructureUncertaintyFunction,
) -> pd.DataFrame:
    """Get the reward data for the nodes in the given tree as dataframe."""
    nodes = [tree.nodes[n] for n in tree.get_leaf_nodes()]
    df = []

    energy_data = reward_func.fetch_adsorption_energy_results(nodes)
    relaxation_error_code = reward_func.fetch_error_codes(nodes)
    reward_data = reward_func.fetch_reward_results(nodes)
    uncertainty_data = uq_func.fetch_uncertainty_results(nodes)
    for n in nodes:
        row = n.return_row()
        row["bulk_composition"] = n.computational_objects["bulk"].formula_pretty
        row["bulk_symmetry"] = n.computational_objects[
            "bulk"
        ].symmetry.crystal_system.value.lower()
        row["site_composition"] = describe_site_placement(
            n.computational_objects["surface"],
            n.computational_params["site_placement"],
        )

        reward_row = reward_data[n._id]
        energy_row = {f"energy_{k}": v for k, v in energy_data[n._id].items()}
        error_code_row = {
            f"error_code_{k}": v for k, v in relaxation_error_code[n._id].items()
        }
        uq_row = {f"uncertainty_{k}": v for k, v in uncertainty_data[n._id].items()}

        row.update(reward_row)
        row.update(energy_row)
        row.update(error_code_row)
        logging.info(error_code_row)
        row.update(uq_row)

        df.append(row)
    return pd.DataFrame(df)


if __name__ == "__main__":
    # df = pd.read_csv("../cu_zn_with_H_uq/reward_values.csv")
    # priors, samples = extract_dft_candidates(df, 100)
    # print(priors)
    # print(len(samples))
    # with open("../cu_zn_with_H_uq/priors.json", "w") as f:
    #     json.dump(priors, f)
    # with open("../cu_zn_with_H_uq/samples.json", "w") as f:
    #     json.dump(samples, f)

    # exit()

    def list_of_strings(arg):
        return arg.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--pathway-file", type=str, default=None)
    parser.add_argument("--attempts", type=int, default=25)
    parser.add_argument("--root-prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--num-bulks", type=int, default=None)
    parser.add_argument("--num-millers", type=int, default=None)
    parser.add_argument("--num-site-compositions", type=int, default=None)

    parser.add_argument("--gnn-model", type=str, default=None)
    parser.add_argument("--gnn-batch-size", type=int, default=None)
    parser.add_argument("--gnn-device", type=str, default=None)
    parser.add_argument("--gnn-ads-tag", type=int, default=None)
    parser.add_argument("--gnn-fmax", type=float, default=None)
    parser.add_argument("--gnn-steps", type=int, default=None)
    parser.add_argument("--gnn-port", type=int, default=None)
    parser.add_argument("--catalyst-symbols", type=list_of_strings)

    args = parser.parse_args()

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for syms in args.catalyst_symbols:
        if syms not in chemical_symbols:
            raise ValueError(f"Unkown chemical symbol {syms}.")

    class TestState:
        def __init__(
            self,
            prompt: str = "Propose a catalyst for the conversion of CO to methanol.",
        ):
            """Initialize self with the given root prompt."""
            assert isinstance(
                prompt, str
            ), f"Prompt is of type {type(prompt)}, not string."
            self.root_prompt = prompt

    # Create the reward function
    # pathways = [
    #     ["*CO", "*COH", "*CHOH", "*CH2OH", "*OHCH3"],
    #     ["*CO", "*CHO", "*CHOH", "*CH2OH", "*OHCH3"],
    # ]
    with open(args.pathway_file, "r") as f:
        pathways = json.load(f)
    # Save the pathways in the data
    with open(save_path / "pathways.json", "w") as f:
        json.dump(pathways, f)

    calc = OCAdsorptionCalculator(
        **{
            "model": args.gnn_model,
            "traj_dir": save_path / "trajectories",
            "batch_size": args.gnn_batch_size,
            "device": args.gnn_device,
            "ads_tag": args.gnn_ads_tag,
            "fmax": args.gnn_fmax,
            "steps": args.gnn_steps,
        }
    )
    reward_func = MicrostructureRewardFunction(
        pathways, calc, num_augmentations_per_site=1, T=args.temperature
    )
    uq_calc = UncertaintyCalculator(
        calc, "data/uq_model_weights/GBMRegressor-peratom_energy.pkl", 0.1, 0.9, 100
    )
    UncertaintyCalculator.traj_dir = save_path / "trajectories"
    uq_func = MicrostructureUncertaintyFunction(
        reaction_pathways=pathways, calc=uq_calc
    )
    # uq_func = UQfunc()

    state = TestState(args.root_prompt)
    # Create the LLM and microstructure planner
    llm_function = AzureOpenaiInterface(dotenv_path=".env", model="gpt-4")
    ms_planner = OCPMicrostructurePlanner(llm_function=llm_function)
    ms_planner.set_state(state)

    if (save_path / "test_node_data.csv").exists() and (
        save_path / "test_edge_data.json"
    ).exists():
        node_data = pd.read_csv(save_path / "test_node_data.csv", index_col=False)
        with open(save_path / "test_edge_data.json", "r") as f:
            edge_data = json.load(f)

        tree = MicrostructureTree.from_data(node_data=node_data, edge_data=edge_data)
        nodes = [tree.nodes[n] for n in tree.get_leaf_nodes()]
        print(len(nodes))

    else:
        attempts = 0
        complete = False
        while not complete:
            try:
                dt = CatalystDigitalTwin()
                syms = args.catalyst_symbols
                dt.computational_params["symbols"] = syms
                dt.computational_objects["symbols"] = syms

                tree = MicrostructureTree(root_node=dt)
                nodes = microstructure_search(tree, ms_planner)
                complete = True
            except Exception as err:
                raise err

        node_data, edge_data, llm_data = tree.store_data(metadata=True)

        # Save to disk
        node_data.to_csv(save_path / "test_node_data.csv", index=False)
        with open(save_path / "test_edge_data.json", "w") as f:
            json.dump(edge_data, f)

        with open(save_path / "llm_answers.json", "w") as f:
            json.dump(llm_data, f)

    print(10 * "" + "finished!" + "*" * 10)

    rewards = reward_func(nodes)
    uq_values = uq_func(nodes)
    for r, u, n in zip(rewards, uq_values, nodes):
        n.set_reward(r)
        n.set_uncertainty(u)

    # Get which nodes to run DFT with
    dft_nodes = microstructure_finetune_selection(
        tree=tree, top_k=4, percentile_reward=0.75
    )

    dft_atoms, dft_names = uq_func.fetch_calculated_atoms(
        [tree.nodes[n] for n in dft_nodes]
    )
    with open(save_path / "structures_for_dft.json", "w") as f:
        json.dump(dft_names, f)
    all_atoms, all_names = uq_func.fetch_calculated_atoms(
        [tree.nodes[n] for n in tree.get_leaf_nodes()]
    )
    dft_dir = save_path / "relaxed_structures"
    dft_dir.mkdir(parents=True, exist_ok=True)
    # Write nodes to disk
    for ats, name in zip(all_atoms, all_names):
        p = dft_dir / (name + ".xyz")
        p.parent.mkdir(parents=True, exist_ok=True)
        write(str(p), ats)

    # Save reward and energy information to disk

    dataframe = get_reward_data(
        tree=tree,
        reward_func=reward_func,
        uq_func=uq_func,
    )
    dataframe.to_csv(save_path / "reward_values.csv", na_rep="NaN")

    print(rewards)

    visualize_tree(tree=tree)
    plt.title("**Placeholder values for rewards and catalyst values**")

    plt.gcf().set_size_inches(18.5, 10.5)
    plt.savefig(save_path / "test_tree.png", dpi=300)
    plt.show()
