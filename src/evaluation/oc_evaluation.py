"""Evaluate the results of oc qeurying tests."""
import pickle
import sys

from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd

sys.path.append("src")
from search.tree_search.atoms.reward import _df_20, _df_22  # noqa: E402
from reasoner.query import parse_answer, generate_oc_prompt  # noqa: E402
from evaluation.visualize import (  # noqa: E402
    unpickle_data,
    search_tree_to_nx,
    remove_colons,
)


def _find_max_location_bfs(given_list):
    max_i = None
    max_j = None
    max_val = None
    for i, sub_list in enumerate(given_list):
        for j, elem in enumerate(sub_list):
            if max_i is None:
                max_i = i
                max_j = j
                max_val = elem
            if elem > max_val:
                max_i = i
                max_j = j
                max_val = elem
    return max_i, max_j, max_val


def read_mcr_to_graph(tree_data):
    """Read the data from a pickle file into a networkx graph."""
    remove_colons(tree_data)
    nx_graph = search_tree_to_nx(tree_data)
    return nx_graph


def _find_max_location_mcr(tree_data):
    graph = read_mcr_to_graph(tree_data)
    max_idx = np.argmax(
        [graph.nodes(data=True)[i]["node_rewards"] for i in range(len(graph.nodes))]
    )
    # print(graph.nodes(data=False)[max_idx])
    print(max_idx)
    sp = nx.shortest_path_length(graph, 0, max_idx)
    return sp


def _limit_depth_mcr(tree_data):
    graph = read_mcr_to_graph(tree_data)
    # print(graph.nodes(data=False)[max_idx])
    sp = dict(nx.shortest_path_length(graph))
    max_depth = max(sp[0].values())
    num_nodes = 0
    for n in graph.nodes():
        if sp[0][n] <= 5:
            num_nodes += 1
    print(f"num_nodes: {num_nodes}")

    return num_nodes, max_depth


def make_combined_df():
    """Make combined df of _oc_20 and _oc_22 datasets."""
    _df_20["e_adsorb"] = _df_20["e_tot"] - _df_20["reference"]
    _df_22["e_adsorb"] = (_df_22["e_tot"] - _df_22["e_slab"]) / _df_22["nads"]
    combined_df = pd.concat([_df_20, _df_22])
    combined_df = combined_df[~combined_df["e_adsorb"].isna()]

    combined_df["e_adsorb"] = combined_df["e_tot"] - combined_df["reference"]
    combined_df = combined_df.sort_values("e_adsorb", ascending=True)
    return combined_df


def parse_oc_data(data_dir: Path):
    """Parse the data out of pickle files in given directory."""
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    adsorbates = pd.concat([_df_20, _df_22])["ads_symbols"].unique()

    results_df = pd.DataFrame()
    api_calls_df = pd.DataFrame()

    max_depths = []

    for ads in adsorbates:
        if ads == "*N":
            continue
        ads = ads.replace("*", "")
        single_shot_path = data_dir / f"single_shot_oc_db_{ads}.pkl"
        multi_shot_paths = [
            data_dir / f"multi_shot_{i}_oc_db_{ads}.pkl" for i in range(10)
        ]
        mcr_path = data_dir / f"mcr_oc_db_{ads}.pkl"
        bfs_path = data_dir / f"bfs_oc_db_{ads}.pkl"
        if any(
            [
                p.exists()
                for p in [single_shot_path, *multi_shot_paths, mcr_path, bfs_path]
            ]
        ):
            if mcr_path.exists():
                # Get search tree energy
                with open(mcr_path, "rb") as f:
                    tree_data = pickle.load(f)
                if len(tree_data["node_rewards"]) > 248:

                    depth = _limit_depth_mcr(tree_data)
                    max_depths.append(depth)
                    results_df.loc["mcr", ads] = max(tree_data["node_rewards"])
                    api_calls_df.loc["mcr", ads] = len(
                        [n["num_queries"] for n in tree_data["nodes"]]
                    )
            if single_shot_path.exists():
                # Get single shot reward
                with open(single_shot_path, "rb") as f:
                    single_state = pickle.load(f)
                results_df.loc["single_shot", ads] = single_state["node_rewards"]
                api_calls_df.loc["single_shot", ads] = single_state["num_queries"]
            if all([p.exists for p in multi_shot_paths]):
                multi_shot_states = []
                for i in range(10):
                    with open(
                        multi_shot_paths[i],
                        "rb",
                    ) as f:
                        current_state = pickle.load(f)
                        multi_shot_states.append(current_state)
                    results_df.loc[f"multi_shot_{i}", ads] = current_state[
                        "node_rewards"
                    ]
                    api_calls_df.loc[f"multi_shot_{i}", ads] = current_state[
                        "num_queries"
                    ]

            if bfs_path.exists():
                # Compare against BFS
                with open(bfs_path, "rb") as f:
                    bfs_tree = pickle.load(f)
                depth_slice = slice(None, -1)
                rewards = [r for sub_l in bfs_tree["node_rewards"] for r in sub_l]
                print(f"BFS_NODES: {len(rewards)}")
                if len(rewards) > 15:
                    results_df.loc["bfs", ads] = max(rewards)
                    api_calls = len(
                        [
                            node["num_queries"]
                            for sub_l in bfs_tree["nodes"][depth_slice]
                            for node in sub_l
                        ]
                    ) + len(
                        [
                            node["num_queries"]
                            for sub_l in bfs_tree["generated_nodes"][depth_slice]
                            for node in sub_l
                        ]
                    )
                    api_calls_df.loc["bfs", ads] = api_calls
                    print(f"BFS_NODES: {api_calls}")
            # print("counting_nodes")
            # tot_nodes = 0
            # for i, subl in enumerate(bfs_tree["nodes"]):
            #     g_subl = bfs_tree["generated_nodes"][i]
            #     print(len(subl) + len(g_subl))
            #     tot_nodes += len(subl) + len(g_subl)
            #     print(tot_nodes)
            #     print("-")
    print(max_depths)
    return results_df, api_calls_df


def best_answers(data_dir: Path):
    """Get MCR best answers."""
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    adsorbates = pd.concat([_df_20, _df_22])["ads_symbols"].unique()

    for ads in adsorbates:
        ads = ads.replace("*", "")
        single_shot_path = data_dir / f"single_shot_oc_db_{ads}.pkl"
        multi_shot_paths = [
            data_dir / f"multi_shot_{i}_oc_db_{ads}.pkl" for i in range(10)
        ]
        mcr_path = data_dir / f"mcr_oc_db_{ads}.pkl"
        bfs_path = data_dir / f"bfs_oc_db_{ads}.pkl"
        if all(
            [
                p.exists()
                for p in [single_shot_path, *multi_shot_paths, mcr_path, bfs_path]
            ]
        ):
            # Get search tree energy
            with open(mcr_path, "rb") as f:
                tree_data = pickle.load(f)

            if len(tree_data["node_rewards"]) > 300:
                max_idx = np.argmax(tree_data["node_rewards"])
                print("----")
                print(tree_data["nodes"][max_idx]["answer"])
                print(tree_data["nodes"][max_idx]["ads_symbols"])
                print("----\n\n\n\n")


if __name__ == "__main__":
    results_df, calls_df = parse_oc_data(Path("post_submission_tests_davinci/oc"))
    print(results_df)
    print(len(results_df.columns))
