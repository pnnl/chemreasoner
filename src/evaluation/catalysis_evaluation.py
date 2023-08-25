"""Evaluate the data from catalysis dataset."""
import pickle
import sys

from pathlib import Path

import pandas as pd
import networkx as nx
import numpy as np


sys.path.append("src")
from evaluation.visualize import (  # noqa: E402
    search_tree_to_nx,
    remove_colons,
)
from llm.query import parse_answer  # noqa: E402


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
    sp = dict(nx.shortest_path_length(graph))
    max_depth = max(sp[0].values())
    # (graph.nodes(data=False)[max_idx])
    sp = nx.shortest_path_length(graph, 0, max_idx)
    return sp, max_depth


def depth_evaluation(directory: Path):
    """Analyze the depth of found maxima."""
    if not isinstance(directory, Path):
        directory = Path(directory)

    for p in directory.rglob("mcr*.pkl"):
        number = str(p).split(".pkl")[0].split("_")[-1]
        with open(p, "rb") as f:
            data = pickle.load(f)
        ()
        # (data["end_time"] - data["start_time"])
        if len(data["node_rewards"]) > 248:
            max(data["node_rewards"])
            sum([n["num_queries"] for n in data["nodes"]])
            (_find_max_location_mcr(data))

    for p in directory.rglob("bfs*input*.pkl"):
        with open(p, "rb") as f:
            data = pickle.load(f)

        # (data["end_time"] - data["start_time"])
        if len([r for sub_l in data["node_rewards"] for r in sub_l]) > 248:
            # max_r = max([r for sub_l in data["node_rewards"] for r in sub_l])
            i, j, val = _find_max_location_bfs(data["node_rewards"])
            (i, j, val)
            (len(data["nodes"]))
            sum(
                [node["num_queries"] for sub_l in data["nodes"] for node in sub_l]
            ) + sum(
                [
                    node["num_queries"]
                    for sub_l in data["generated_nodes"]
                    for node in sub_l
                ]
            )
            ("counting_nodes")
            tot_nodes = 0
            for i, subl in enumerate(data["nodes"]):
                g_subl = data["generated_nodes"][i]
                (len(subl))
                (len(g_subl))
                (len(subl) + len(g_subl))
                tot_nodes += len(subl) + len(g_subl)
                (tot_nodes)
                ("-")


def parse_catalysis_data(directory: Path):
    """Read in the data in given directory."""
    if not isinstance(directory, Path):
        directory = Path(directory)

    max_depths = []
    multi_shot_answers = []

    results_df = pd.DataFrame()
    api_calls_df = pd.DataFrame()
    for p in directory.rglob("single_shot*.pkl"):
        number = str(p).split(".pkl")[0].split("input_")[-1]
        (number)
        with open(p, "rb") as f:
            data = pickle.load(f)

        if number not in ["db", "48", "49"]:
            results_df.loc["single_shot", number] = data["node_rewards"]
            api_calls_df.loc["single_shot", number] = data["num_queries"]

    for p in directory.rglob("multi_shot*.pkl"):
        number = str(p).split(".pkl")[0].split("input_")[-1]
        with open(p, "rb") as f:
            data = pickle.load(f)

        if number not in ["db", "48", "49"]:
            multi_shot_number = str(p).split("multi_shot_")[-1].split("_")[0]
            results_df.loc[f"multi_shot_{multi_shot_number}", number] = data[
                "node_rewards"
            ]
            api_calls_df.loc[f"multi_shot_{multi_shot_number}", number] = data[
                "num_queries"
            ]
            multi_shot_answers.append(tuple(parse_answer(data["answer"])))

    for p in directory.rglob("mcr*.pkl"):
        number = str(p).split(".pkl")[0].split("input_")[-1]
        with open(p, "rb") as f:
            data = pickle.load(f)
        ()
        (data["end_time"] - data["start_time"])
        if len(data["node_rewards"]) > 248:
            results_df.loc["mcr", number] = max(data["node_rewards"])
            api_calls_df.loc["mcr", number] = len(
                [n["num_queries"] for n in data["nodes"]]
            )
            loc, max_depth = _find_max_location_mcr(data)
            max_depths.append(max_depth)

    for p in directory.rglob("bfs*.pkl"):

        if "input" in str(p):
            (p)
            with open(p, "rb") as f:
                data = pickle.load(f)
            number = str(p).split(".pkl")[0].split("input_")[-1]
            number = number[:-1] + number[-1].zfill(2)
            (data["end_time"] - data["start_time"])
            if len([r for sub_l in data["node_rewards"] for r in sub_l]) > 2:
                (len([r for sub_l in data["node_rewards"] for r in sub_l]))
                results_df.loc["bfs", number] = max(
                    [r for sub_l in data["node_rewards"][:-1] for r in sub_l]
                )
                api_calls_df.loc["bfs", number] = len(
                    [node["num_queries"] for sub_l in data["nodes"] for node in sub_l]
                ) + len(
                    [
                        node["num_queries"]
                        for sub_l in data["generated_nodes"]
                        for node in sub_l
                    ]
                )
    (max_depths)
    return results_df, api_calls_df


if __name__ == "__main__":
    print(
        parse_catalysis_data(
            "/Users/spru445/alchemist/post_submission_tests_davinci/biofuel"
        )[0]
    )
    # depth_evaluation("/Users/spru445/alchemist/post_submission_tests_davinci/biofuel")
