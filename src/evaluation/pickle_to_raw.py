"""Take pickle file output and return networkx graphs."""
import copy
import json
import pickle
import sys

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

sys.path.append("src")
from evaluation.visualize import (  # noqa: E402
    unpickle_data,
    search_tree_to_nx,
    remove_colons,
    nx_plot,
)  # noqa: E402


def clean_attr_data(attribute_data):
    """Clean attribute data so it doesn't trip ':' issue."""
    if isinstance(attribute_data, str) and ":" in attribute_data:
        return '"' + attribute_data + '"'
    elif isinstance(attribute_data, list):
        return [clean_attr_data(attr) for attr in attribute_data]
    elif isinstance(attribute_data, dict):
        return {
            clean_attr_data(k): clean_attr_data(v) for k, v in attribute_data.items()
        }
    else:
        return attribute_data


def read_single_query_to_json(pickle_file: Path):
    """Read in a single query to a json object."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def read_mcr_to_graph(pickle_file: Path):
    """Read the data from a pickle file into a networkx graph."""
    data = unpickle_data(pickle_file)
    remove_colons(data)
    nx_graph = search_tree_to_nx(data)
    return nx_graph


def bfs_to_nx(tree_data):
    """Make an nx graph from bfs data."""
    nodes_attr = []
    nodes_list = []
    node_indices = []
    node_counter = 0
    for i, node_list in enumerate(tree_data["nodes"]):
        added_indices = []
        for j, n in enumerate(node_list):
            added_indices.append(node_counter)
            nodes_list.append(node_counter)
            n["node_rewards"] = tree_data["node_rewards"][i][j]
            nodes_attr.append(clean_attr_data(n))
            node_counter += 1
        node_indices.append(added_indices)

    edge_list = []

    for i, parent_list in enumerate(tree_data["parent_idx"]):
        for j, parent_idx in enumerate(parent_list):
            current_node = node_indices[i][j]
            if parent_idx != -1:
                parent_node = node_indices[i - 1][parent_idx]
                edge_list.append((parent_node, current_node))

    for i, gen_node_list in enumerate(tree_data["generated_nodes"]):
        parent_list = tree_data["generated_parent_idx"][i]
        for j, gen_node in enumerate(gen_node_list):
            nodes_list.append(node_counter)
            gen_node["node_rewards"] = tree_data["generated_node_rewards"][i][j]
            nodes_attr.append(clean_attr_data(gen_node))

            parent_idx = parent_list[j]
            parent_node = node_indices[i - 1][parent_idx]
            edge_list.append((parent_node, node_counter))

            node_counter += 1

    G = nx.Graph(edge_list)
    nodes_dict = {idx: nodes_attr[i] for i, idx in enumerate(nodes_list)}
    nx.set_node_attributes(G, nodes_dict)
    nx.set_edge_attributes(G, 1, name="log_action")
    return G


def read_bfs_to_graph(pickle_file: Path):
    """Read the data from a pickle file into a networkx graph."""
    data = unpickle_data(pickle_file)
    print(len(data["nodes"]))
    for k, v in data["nodes"][1][0].items():
        print(f"{k}:\t{type(v)}")
    nx_graph = bfs_to_nx(data)
    return nx_graph


def add_atribute(nx_graph, node_func, name):
    """Add an attribute to the graph using node function to determine value."""
    attr_dict = dict()
    i = 0
    for n, data in nx_graph.nodes.items():
        i += 1
        val = node_func(data)
        attr_dict[n] = val
    nx.set_node_attributes(nx_graph, attr_dict, name=name)


def segment_list_by_quotes(char_list: list[str], quotechar="'"):
    """Segment list into sections between quotes."""
    lists = []
    current_list = []
    found_quotechar = False
    for char in char_list:
        if char == "'":
            if len(current_list) == 0:
                found_quotechar = True
            else:
                found_quotechar = False
                lists.append("".join(current_list))
                current_list = []
        elif found_quotechar:
            current_list.append(char)
    return lists


def parse_string_list(string_list):
    """Parse string represenation of lsit into an actual list."""
    string_list = string_list.replace(",", "", 1)
    string_list = string_list.replace("[", "").strip()
    print("\n\n\n\n\n\n")
    print(segment_list_by_quotes(string_list.split(", ")))
    print("\n\n\n\n\n\n")
    return [inc for inc in string_list.split(", ") if inc.strip() not in ["", "[", "]"]]


def create_prompt(node_data, split=True):
    """Create the prompt from node_data."""
    parsed_node_data = copy.deepcopy(node_data)
    if "model" in parsed_node_data.keys():
        parsed_node_data.pop("model")
    print(parsed_node_data)
    if split:
        for k in [
            "include_list",
            "exclude_list",
            "prev_candidate_list",
            "ads_symbols",
            "ads_preferences",
        ]:
            if not isinstance(node_data[k], list):
                parsed_node_data[k] = parse_string_list(node_data[k])
        print(dict(**parsed_node_data))
        qs = QueryState(**parsed_node_data)
        return qs.prompt
    else:
        qs = QueryState(**parsed_node_data)
        return qs.prompt


def create_level_data(G):
    """Calculate the distance from the root node for each node and store it."""
    sp = dict(spl=dict(nx.all_pairs_shortest_path_length(G)))
    distances_dict = dict()
    for n in G.nodes():
        distance = sp[0][n]
        distances_dict[n] = distance
    nx.set_node_attributes(G, distances_dict, name="level")


def create_current_candidate_list(node_data):
    """Create the prompt from node_data."""
    if node_data["answer"] is not None:
        answer_list = parse_answer(node_data["answer"])
        return answer_list
    else:
        return None


def create_prompt_from_node_data(node_data):
    """Create a prompt from node data in a graph."""
    qs = QueryState.from_dict(node_data)
    return qs.prompt


def graph_get_trace(graph: nx.Graph):
    """Dump the trace to the best node in a graph."""
    max_idx = np.argmax(
        [graph.nodes(data=True)[i]["node_rewards"] for i in range(len(graph.nodes))]
    )
    # print(graph.nodes(data=False)[max_idx])
    sp = nx.all_simple_paths(graph, 0, max_idx)

    if len(list(nx.cycle_basis(graph))) > 0:
        nx_plot(graph)
        plt.show()
        raise ValueError("A graph has a cycle!")

    messages = []
    print(list(sp))
    if len(list(sp)) > 0:
        for node in list(sp)[0]:
            prompt = create_prompt_from_node_data(graph.nodes()[node])

            answer = graph.nodes()[node]["answer"]

            messages.append(" * " + str(graph.nodes()[node]["node_rewards"]))
            messages.append("")
            messages.append("P:\n" + prompt)
            messages.append("A:\n" + str(answer))
            messages.append("-" * 80 + "\n\n")

    return messages


with open("all_adsorption_energies.json", "r") as f:
    adsorption_energy_data = json.load(f)

search_results = pd.DataFrame(
    columns=["llm", "method", "policy", "reward_function", "best_reward", "query"]
)
data_for_plots = []
with open("iclr_traces.txt", "w"):  # clear the file
    pass
usage_statistics = []
if __name__ == "__main__":
    for llm in ["gpt", "llama"]:
        for p in Path("data", "output", f"iclr_{llm}_timing", "").rglob("*.pkl"):
            file_name = str(p).split("reward_")[-1].split(".")[0]
            method = p.stem.split("_")[0]
            policy = p.stem.split(method + "_")[-1].split("_")[0]
            reward = p.stem.split(policy + "_")[-1].split("_")[0]

            try:
                with open(p, "rb") as f:
                    runs_data = pickle.load(f)
            except Exception:
                continue

            for i, data_entry in enumerate(runs_data):
                trace_messages = []
                if "single" in str(p):
                    search_results.to_csv(Path("data", "output", "search_results.csv"))
                    if isinstance(data_entry, tuple):
                        tree_data, err, trace = data_entry
                    else:
                        tree_data, err, trace = (data_entry, "", "")
                    print(tree_data)
                    # print(p)
                    # print(type(tree_data["node_rewards"]))
                    # print(tree_data["info"]["generation"]["candidates_list"])
                    # print(tree_data["info"]["simulation-reward"])
                    # print(tree_data["info"]["llm-reward"])
                    for i, llm_results in enumerate(
                        tree_data["info"]["llm-reward"]["attempted_prompts"][-1][
                            "number_answers"
                        ]
                    ):  # Assume candidate catalysts are same order after llm gen
                        for j, candidate in enumerate(
                            tree_data["info"]["generation"]["candidates_list"]
                        ):
                            slab_syms = tree_data["info"]["simulation-reward"][
                                "slab_syms"
                            ][j]
                            if slab_syms is not None:
                                print(slab_syms)
                                ads_syms = tree_data["ads_symbols"][i]
                                adslab_syms = "".join(slab_syms) + "_" + ads_syms
                                simulation_value = adsorption_energy_data[adslab_syms][
                                    "adsorption_energy"
                                ]
                                scatterplot_data = {
                                    "llm_value": llm_results[j],
                                    "simulation_value": simulation_value,
                                    "candidate": candidate,
                                    "candidates_symbols": slab_syms,
                                    "adsorbates": ads_syms,
                                    "adslab": adslab_syms,
                                }
                                data_for_plots.append(scatterplot_data)

                                llm_reward_table_data = {
                                    "llm": llm,
                                    "method": "zero-shot",
                                    "policy": "zero-shot",
                                    "reward_function": "llm-reward",
                                    "best_reward": tree_data["info"]["llm-reward"][
                                        "value"
                                    ],
                                    "query": (file_name, i),
                                }
                                sim_reward_table_data = {
                                    "llm": llm,
                                    "method": "zero-shot",
                                    "policy": "zero-shot",
                                    "reward_function": "simulation-reward",
                                    "best_reward": tree_data["info"][
                                        "simulation-reward"
                                    ]["value"],
                                    "query": (file_name, i),
                                }
                                search_results = pd.concat(
                                    [
                                        search_results,
                                        pd.DataFrame(llm_reward_table_data),
                                    ]
                                )
                                search_results = pd.concat(
                                    [
                                        search_results,
                                        pd.DataFrame(sim_reward_table_data),
                                    ]
                                )

                else:
                    tree_data, err, trace = data_entry

                    if "beam-search" in str(p):
                        graph = bfs_to_nx(tree_data)

                    elif "mcts" in str(p):
                        graph = search_tree_to_nx(tree_data)

                    if len(graph.nodes) > 1:
                        usage_statistics.append(
                            {
                                "filename": str(p),
                                "llm_count": 0,
                                "llm_time": 0,
                                "llm_avg": None,
                                "prompt_length": 0,
                                "answer_length": 0,
                                "gnn_count": 0,
                                "gnn_time": 0,
                                "gnn_avg": None,
                            }
                        )
                        for n, data in graph.nodes(data=True):
                            if "simulation-reward" in data["info"].keys():
                                usage_statistics[-1]["llm_count"] += data["num_queries"]
                                usage_statistics[-1]["llm_time"] += data["query_time"]
                                usage_statistics[-1]["llm_avg"] = (
                                    usage_statistics[-1]["llm_time"]
                                    / usage_statistics[-1]["llm_count"]
                                )
                                usage_statistics[-1]["gnn_count"] += data["info"][
                                    "simulation-reward"
                                ]["gnn_calls"]
                                usage_statistics[-1]["gnn_time"] += data["info"][
                                    "simulation-reward"
                                ]["gnn_time"]
                                if usage_statistics[-1]["gnn_time"] != 0:
                                    usage_statistics[-1]["gnn_avg"] = (
                                        usage_statistics[-1]["gnn_time"]
                                        / usage_statistics[-1]["gnn_count"]
                                    )
                                print(list(data["info"]["generation"].keys()))
                                usage_statistics[-1]["prompt_length"] += len(
                                    data["info"]["generation"]["prompt"]
                                ) + len(data["info"]["generation"]["system_prompt"])
                                usage_statistics[-1]["answer_length"] += len(
                                    data["info"]["generation"]["answer"]
                                )
                            else:
                                print(list(data["info"].keys()))

                        usage_statistics[-1]["avg_prompt_length"] = usage_statistics[
                            -1
                        ]["prompt_length"] / len(graph.nodes)
                        usage_statistics[-1]["avg_answer_length"] = usage_statistics[
                            -1
                        ]["answer_length"] / len(graph.nodes)
                        usage_statistics[-1]["overall_time"] = (
                            tree_data["end_time"] - tree_data["start_time"]
                        )
                        usage_statistics[-1]["num_nodes"] = len(graph.nodes)
                        usage_statistics[-1]["avg_node_time"] = (
                            usage_statistics[-1]["overall_time"]
                            / usage_statistics[-1]["num_nodes"]
                        )

                        trace_messages.append(p.stem + f"_{i}\n\n")
                        trace_messages += graph_get_trace(graph)
                        trace_messages.append(
                            "\n\n" + "=" * 80 + "\n" + "=" * 80 + "\n\n"
                        )
                        with open("data/output/search_traces.txt", "a") as f:
                            f.write("\n".join(trace_messages))

                        data = {
                            "llm": llm,
                            "method": p.stem.split("_")[0],
                            "policy": p.stem.split(method + "_")[-1].split("_")[0],
                            "reward_function": p.stem.split(policy + "_")[-1].split(
                                "_"
                            )[0],
                            "best_reward": max(
                                nx.get_node_attributes(graph, "node_rewards").values()
                            ),
                            "query": (file_name, i),
                        }
                        print(data)
                        search_results = pd.concat(
                            [
                                search_results,
                                pd.DataFrame.from_dict(data, orient="index").T,
                            ],
                            ignore_index=True,
                        )
                        print(err)
                        print(trace)
                        print(str(p))
    with open("data/output/scatterplot_data.json", "w") as f:
        pass
        # json.dump(data_for_plots, f)
    # print(search_results)
    # search_results.to_csv(Path("data", "output", "search_results.csv"))

    with open("data/output/usage_statistics.json", "w") as f:
        json.dump(usage_statistics, f)
