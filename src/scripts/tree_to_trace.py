"""Turn a directory of search trees into traces."""
import argparse
import json
import os

from pathlib import Path

import networkx as nx
import numpy as np


def get_max_adsorption_energies(node):
    if "simulation-reward" in node["info"].keys():
        reward_values = node["info"]["simulation-reward"]["reward_values"]
        numbers = []
        for cat, v in reward_values.items():
            print(cat)
            for ads, values in v.items():
                print(ads)
                numbers.append(min(values))
        if len(numbers) > 0:
            return max(numbers)
        else:
            return node["node_rewards"]
    else:
        return node["node_rewards"]


parser = argparse.ArgumentParser()

parser.add_argument("--processed-dir", type=str)
parser.add_argument("--traces-dir", type=str)
parser.add_argument("--max-reward", type=float)

args = parser.parse_args()
processed_dir = Path(args.processed_dir)
traces_dir = Path(args.traces_dir)
max_reward = args.max_reward

tree_rewards = {}

for p in processed_dir.rglob("*.json"):
    if str(p.parent) not in tree_rewards.keys():
        tree_rewards[str(p.parent)] = []
    print(p)

    # Have to filter for nodes with rewards less than a max threshold. This is because
    # some catalysts in have inconsistent lattic structure between ASE and materials
    # Project (i.e. Ba, Be, Sr). This is becomes an issue when doing weak adsorption questions
    # and adsorption pathways questions. Wehen finding the max node, values are filtered
    # to be lower than max_reward.

    with open(str(p), "r") as f:
        data = json.load(f)

    graph = nx.readwrite.json_graph.tree_graph(data)

    max_idx = np.argmax(
        [
            graph.nodes(data=True)[i]["node_rewards"]
            if get_max_adsorption_energies(graph.nodes(data=True)[i]) < max_reward
            else -np.inf
            for i in range(len(graph.nodes))
        ]
    )
    print(
        [graph.nodes(data=True)[i]["node_rewards"] for i in range(len(graph.nodes))][
            max_idx
        ]
    )
    filtered_list = [
        (i, get_max_adsorption_energies(graph.nodes(data=True)[i]))
        for i in range(len(graph.nodes))
        if get_max_adsorption_energies(graph.nodes(data=True)[i]) > max_reward
    ]

    if len(filtered_list) > 0:
        print(f"\nFiltered out nodes: {filtered_list}\n")

    sp = list(nx.all_simple_paths(graph, 0, max_idx))
    # sp =[[1,2,3,4]]
    print(sp)
    output_nodes = []
    if len(sp) != 0:
        print(sp)
        print(len(sp))
        for node in sp[0]:
            output_nodes.append(graph.nodes(data=True)[node])

    (traces_dir / p).parent.mkdir(parents=True, exist_ok=True)
    with open(traces_dir / p, "w") as f:
        json.dump(output_nodes, f, indent=4)
    best_output_path = traces_dir / (p.parent / (p.stem + ".best.json"))
    if len(output_nodes) > 0:
        with open(best_output_path, "w") as f:
            json.dump(output_nodes[-1], f, indent=4)

    if len(output_nodes) > 0:
        tree_rewards[str(p.parent)] = output_nodes[-1]["node_rewards"]
    else:
        tree_rewards[str(p.parent)] = np.nan

for k, v in tree_rewards.items():
    print(f"{k}: {np.mean(v)}")
