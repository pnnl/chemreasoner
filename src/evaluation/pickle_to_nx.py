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
from llm.query import parse_answer, QueryState  # noqa: E402


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


if __name__ == "__main__":
    for p in Path(
        "/Users/spru445/alchemist/data/output_data/post_submission_tests_davinci"
    ).rglob("mcr*.pkl"):
        continue
        graph = read_mcr_to_graph(p)
        if len(graph.nodes()) >= 200:

            # add_atribute(graph, create_prompt, name="prompt")
            # add_atribute(
            #     graph, create_current_candidate_list, name="current_candidates"
            # )
            # add_atribute(graph, create_embedding, name="embedding")

            single_path = Path(str(p).replace("mcr", "single_shot"))
            if single_path.exists():
                single_json = read_single_query_to_json(single_path)
                single_json["prompt"] = create_prompt(single_json, split=False)
                single_json["current_candidates"] = create_current_candidate_list(
                    single_json
                )
                # single_json["embedding"] = create_embedding(single_json)
                with open(str(single_path).replace(".pkl", ".json"), "w") as f:
                    json.dump(single_json, f)

            multi_shot = []
            for i in range(10):
                multi_path = Path(str(p).replace("mcr", f"multi_shot_{i}"))
                if multi_path.exists():
                    multi_json = read_single_query_to_json(multi_path)
                    multi_json["prompt"] = create_prompt(multi_json, split=False)
                    multi_json["current_candidates"] = create_current_candidate_list(
                        multi_json
                    )
                    # multi_json["embedding"] = create_embedding(multi_json)
                    multi_shot.append(multi_json)

            with open(
                str(multi_path)
                .replace(f"multi_shot_{i}", "multi_shot")
                .replace(".pkl", ".json"),
                "w",
            ) as f:
                json.dump(multi_shot, f)

            # nx.write_gpickle(graph, str(p).replace(".pkl", ".gpickle"))

    for p in Path("/Users/spru445/alchemist/post_submission_tests_davinci").rglob(
        "bfs*.pkl"
    ):
        continue
        graph = read_bfs_to_graph(p)

        add_atribute(graph, create_prompt, name="prompt")
        add_atribute(graph, create_current_candidate_list, name="current_candidates")
        # add_atribute(graph, create_embedding, name="embedding")
        nx.write_gpickle(graph, str(p).replace(".pkl", ".gpickle"))

    i = 0
    found_oc = False
    found_biofuels = False
    queries = []
    search_methods = []
    datasets = []
    rewards = []
    prompts = []
    raw_answers = []
    all_answers = []
    for p in list(
        Path(
            "/Users/spru445/alchemist/data/output_data/post_submission_tests_davinci/"
        ).rglob("*mcr*.gpickle")
    ) + list(
        Path(
            "/Users/spru445/alchemist/data/output_data/post_submission_tests_davinci/"
        ).rglob("*bfs*.gpickle")
    ):
        if "bfs" in str(p):
            bfs_p = p
            # for bfs_p, mcr_p in hardcoded_paths:
            bfs_G = nx.read_gpickle(bfs_p)
            # nx_plot(bfs_G, title=f"{bfs_p.stem}")
            # plt.savefig(Path("traces") / f"{bfs_p.stem}.pdf")
            max_idx = np.argmax(
                [
                    bfs_G.nodes(data=True)[i]["node_rewards"]
                    for i in range(len(bfs_G.nodes))
                ]
            )
            sp = nx.all_simple_paths(bfs_G, 0, max_idx)
            messages = []
            for node in list(sp)[0]:
                messages.append("-------")
                messages.append(bfs_G.nodes()[node]["node_rewards"])
                messages.append(bfs_G.nodes()[node]["answer"])
                messages.append(bfs_G.nodes()[node]["prompt"])
                messages.append("-------")

                with open(Path("traces") / f"{bfs_p.stem}.txt", "w") as f:
                    f.writelines([str(mes) + "\n" for mes in messages])

            for node in bfs_G.nodes():
                datasets.append("oc" if "oc" in bfs_p.stem else "biofuels")
                if "oc" in bfs_p.stem:
                    queries.append(bfs_p.stem.split("_")[-1])
                elif "biofuels" in bfs_p.stem:
                    queries.append("input" + bfs_p.stem.split("input")[-1])
                else:
                    raise ValueError(f"Unkown dataset for file {bfs_p}.")
                search_methods.append("beam_search")
                rewards.append(bfs_G.nodes()[node]["node_rewards"])
                prompts.append(bfs_G.nodes()[node]["prompt"])
                raw_answers.append(bfs_G.nodes()[node]["answer"])
                all_answers.append(
                    parse_answer(
                        bfs_G.nodes()[node]["answer"]
                        if bfs_G.nodes()[node]["answer"] is not None
                        else ""
                    )
                )

        if "mcr" in str(p):
            mcr_p = p
            mcr_G = nx.read_gpickle(mcr_p)
            # nx_plot(mcr_G, title=f"{mcr_p.stem}")
            # plt.savefig(Path("traces") / f"{mcr_p.stem}.pdf")
            max_idx = np.argmax(
                [
                    mcr_G.nodes(data=True)[i]["node_rewards"]
                    for i in range(len(mcr_G.nodes))
                ]
            )
            sp = nx.all_simple_paths(mcr_G, 0, max_idx)
            messages = []

            for node in list(sp)[0]:
                messages.append("-------")
                messages.append(mcr_G.nodes()[node]["node_rewards"])
                messages.append(mcr_G.nodes()[node]["answer"])
                messages.append(mcr_G.nodes()[node]["prompt"])
                messages.append("-------")
                # print(mcr_G.nodes()[node]["node_rewards"])
                # print(mcr_G.nodes()[node]["prompt"])
                # print(mcr_G.nodes()[node]["answer"])

            for node in mcr_G.nodes():
                datasets.append("oc" if "oc" in mcr_p.stem else "biofuels")
                if "oc" in mcr_p.stem:
                    queries.append(mcr_p.stem.split("_")[-1])
                elif "biofuels" in mcr_p.stem:
                    queries.append("input" + bfs_p.stem.split("input")[-1])
                else:
                    raise ValueError(f"Unkown dataset for file {bfs_p}.")
                search_methods.append("mcts")
                rewards.append(mcr_G.nodes()[node]["node_rewards"])
                prompts.append(mcr_G.nodes()[node]["prompt"])
                raw_answers.append(mcr_G.nodes()[node]["answer"])
                all_answers.append(
                    parse_answer(
                        mcr_G.nodes()[node]["answer"]
                        if mcr_G.nodes()[node]["answer"] is not None
                        else ""
                    )
                )

        break

    for p in Path(
        "/Users/spru445/alchemist/data/output_data/post_submission_tests_davinci/"
    ).rglob("*shot*.json"):
        with open(p, "r") as f:
            data = json.load(f)
        if "multi" in str(p):
            for obj in data:
                datasets.append("oc" if "oc" in p.stem else "biofuels")
                if "oc" in mcr_p.stem:
                    queries.append(mcr_p.stem.split("_")[-1])
                elif "biofuels" in mcr_p.stem:
                    queries.append("input" + bfs_p.stem.split("input")[-1])
                search_methods.append("multi_shot")
                rewards.append(obj["node_rewards"])
                prompts.append(obj["prompt"])
                raw_answers.append(obj["answer"])
                all_answers.append(
                    parse_answer(obj["answer"] if obj["answer"] is not None else "")
                )
        else:
            datasets.append("oc" if "oc" in p.stem else "biofuels")
            if "oc" in mcr_p.stem:
                queries.append(mcr_p.stem.split("_")[-1])
            elif "biofuels" in mcr_p.stem:
                queries.append("input" + bfs_p.stem.split("input")[-1])
            search_methods.append("single_shot")
            rewards.append(data["node_rewards"])
            prompts.append(data["prompt"])
            raw_answers.append(data["answer"])
            all_answers.append(
                parse_answer(data["answer"] if data["answer"] is not None else "")
            )

        # with open(Path("traces") / f"{mcr_p.stem}.txt", "w") as f:
        #     f.writelines([str(mes) + "\n" for mes in messages])
    all_answers = [ans_list + [None] * (5 - len(ans_list)) for ans_list in all_answers]
    max_length = len(max(all_answers, key=len))
    data_df = pd.DataFrame(all_answers, columns=[f"ans_{i}" for i in range(max_length)])
    data_df["reward"] = rewards
    data_df["search_method"] = search_methods
    data_df["dataset"] = datasets
    data_df["query"] = queries
    data_df["prompt"] = prompts
    data_df["raw_answer"] = raw_answers
    print(data_df.head())
    data_df.to_csv("llm_answers.csv", index=False)

    plt.show()
