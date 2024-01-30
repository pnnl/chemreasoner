#!/usr/bin/python


import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

_category_mapper = {
    "OpenCatalyst": "OpenCatalyst",
    "BioFuels": "BioFuels",
    "RWGS": "BioFuels",
    "CO2ToEthanol": "CO2-Fuel",
    "CO2ToMethanol": "CO2-Fuel",
}


class Simtree:
    def __init__(self, root, sim_name, sim_type):
        self.root = root
        self.sim_name = sim_name
        self.sim_type = sim_type
        self.parent_dict = dict()
        self.prompt_dict = dict()
        self.reward_dict = dict()
        self.is_valid = dict()
        self.set_per_node_data(tmp_node=self.root, parent_id=-100)
        self.max_reward, self.max_reward_node = self.get_max_reward_node(self.root)

        reward_path = self.get_path_to_root(self.max_reward_node)
        # print(self.max_reward, self.max_reward_node, reward_path)
        self.max_reward_path = [self.max_reward_node] + reward_path[0:-1]
        # print(self.sim_name,  self.max_reward, self.max_reward_node,  self.max_reward_path)

    def set_per_node_data(self, tmp_node, parent_id):
        tmp_node["parent_id"] = parent_id

        self.parent_dict[tmp_node["id"]] = parent_id
        self.prompt_dict[tmp_node["id"]] = tmp_node["generation_prompt"]
        self.reward_dict[tmp_node["id"]] = tmp_node["node_rewards"]
        self.is_valid[tmp_node["id"]] = self.get_node_validity(tmp_node)
        children = self.get_children(tmp_node)
        for c in children:
            self.set_per_node_data(c, tmp_node["id"])

    def get_children(self, node):
        """
        returns a list containing child nodes of the input node \
        or empty if leaf node
        """
        if "children" in node.keys():
            return node["children"]
        else:
            return []

    def get_path_to_root(self, node_id):
        path = []
        if node_id in self.parent_dict:
            parent = self.parent_dict[node_id]
            path.append(parent)
            path = path + self.get_path_to_root(parent)
        return path

    def get_max_reward_node_old(self, root):
        # ignore invalid nodes
        # if(root['is_valid'] == False):
        #   return -np.inf, root['id']

        root_reward = root["node_rewards"]
        children = self.get_children(root)
        if (len(children)) == 0:
            return root_reward, root["id"]

        max_reward = root_reward
        max_reward_node = root["id"]
        max_reward_path = []
        for c in children:
            # tmp_reward, tmp_reward_node, tmp_path = get_max_reward_node(c, current_path)
            tmp_reward, tmp_reward_node = self.get_max_reward_node(c)
            if tmp_reward > max_reward:
                max_reward = tmp_reward
                max_reward_node = tmp_reward_node
                # max_reward_path = tmp_path
        return max_reward, max_reward_node

    def get_max_reward_node(self, root):
        max_id = root["id"]
        max_reward = root["node_rewards"]
        for node_id, node_reward in self.reward_dict.items():
            if self.is_valid[node_id] and (node_reward > max_reward):
                max_reward = node_reward
                max_id = node_id
        return max_reward, max_id

    def print_tree(self, node):
        children = self.get_children(node)
        if (len(children)) == 0:
            print(str(node["id"]) + "[p:" + str(node["parent_id"]) + "] --> NA")
            return
        else:
            cids = " ".join(str(c["id"]) for c in children)
            print(str(node["id"]) + "[p:" + str(node["parent_id"]) + "] -->", cids)
            for c in children:
                self.print_tree(c)

    def create_path_embeddings(self, model, path):
        """
        given as input a dictionary containing
        1) {nodeid : generation_prompt} and
        2) a language embedding model
        return a
        dictionary {node_id :generation_prompt_embedding}
        """
        embeddings = dict()
        for nodeid in path:
            node_prompt = self.prompt_dict[nodeid]
            node_emb = model.encode(node_prompt, convert_to_tensor=True)
            embeddings[nodeid] = node_emb
        return embeddings

    def get_path_coherence(self, model):
        """
        Returns minimum similarity betweenany two
        consecutive pairs of nodes in a path
        """
        path = self.max_reward_path
        node_embeddings = self.create_path_embeddings(model, path)
        path_embedding = [node_embeddings[n] for n in path]
        # print(len(path_embedding), path_embedding[0])
        min_sim = np.inf
        for i in range(len(path_embedding) - 1):
            emb1 = path_embedding[i]
            emb2 = path_embedding[i + 1]
            # print(emb1, emb2)
            pair_sim = util.pytorch_cos_sim(emb1, emb2)
            # print(pair_sim)
            if pair_sim < min_sim:
                min_sim = pair_sim
        min_sim = float(min_sim)
        self.coherence = min_sim
        return min_sim

    def get_metrics(self, model):
        metrics = dict()
        metrics["sim_name"] = self.sim_name
        metrics["sim_type"] = self.sim_type
        metrics["max_reward"] = self.max_reward
        metrics["max_reward_node"] = self.max_reward_node
        metrics["max_reward_path"] = self.max_reward_path
        metrics["max_reward_hops"] = len(self.max_reward_path) - 1
        metrics["max_reward_path_coherence"] = self.get_path_coherence(model)
        return metrics

    def get_node_validity(self, node):
        # a node reward is only valid if this number is < 10
        reward = node["node_rewards"]
        if "simulation-reward" in node["info"].keys():
            reward_values = node["info"]["simulation-reward"]["reward_values"]
            numbers = []
            for cat, v in reward_values.items():
                # print(cat)
                for ads, values in v.items():
                    # print(ads)
                    numbers.append(min(values))
            if len(numbers) > 0:
                reward = max(numbers)
        return reward < 10


def load_sim_trees(datadir):
    tree_objs = dict()
    print("Loadfing Simulation data from ", datadir)
    for filename in os.listdir(datadir):
        filepath = datadir + filename
        data = json.load(open(filepath))
        tree_objs[filename] = data
    return tree_objs


def init_sim_trees(tree_objs, query_type_file="query_to_type_dataset.csv"):
    sim_trees = []
    query_types = pd.read_csv(query_type_file)
    query_types = list(query_types["dataset"].map(lambda x: _category_mapper[x]).values)

    for k, v in tree_objs.items():
        """K=search_tree_26.json,"""
        # print("Processing simulation ", k)
        sim_id = int(k[12:-5])
        # print(sim_id)
        s = Simtree(root=v, sim_name=k, sim_type=query_types[sim_id])
        sim_trees.append(s)
    return sim_trees


def get_sim_results(maindir, strategy_name, model, outdir):
    datadir = maindir + "/" + strategy_name + "/"
    tree_objs = load_sim_trees(datadir)
    print("Finished loading data")
    print("Processing the simulations")
    sim_trees = init_sim_trees(tree_objs)
    metrics = list()
    for tree in sim_trees:
        m = tree.get_metrics(model)
        metrics.append(m)
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(outdir + strategy_name + "_results.csv")
    return df


def print_sim_metrics(df, strategy_name):
    sim_types = df["sim_type"].unique()
    rewards = "Avg_reward, "
    depths = "Avg_depth, "
    coherence = "Avg_coherence, "
    # print(sim_types)

    for sim_type in sim_types:
        df_q = df[df["sim_type"] == sim_type]
        q_rewards = np.average(list(df_q["max_reward"].values))
        q_depth = np.average(list(df_q["max_reward_hops"].values))
        q_coherence = np.average(list(df_q["max_reward_path_coherence"].values))
        rewards += str(q_rewards) + ", "
        depths += str(q_depth) + ", "
        coherence += str(q_coherence) + ", "
        # answer[sim_type] = {'avg_reward': q_rewards, \
        #                    'avg_depth':q_depth, 'avg_coherence':q_coherence}
    str_sim_types = ", ".join([x for x in sim_types])
    print(strategy_name + ", " + str_sim_types)
    print(rewards)
    print(depths)
    print(coherence)
    # print(answer)
    return


# NOte that this doesnt work if VPN is connected, disconnect VPN
##For OpenAI model
# ChemReasoner-Canada-text-embedding-ada-002
if __name__ == "__main__":
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    maindir = "icml_processed/"
    strategy_names = os.listdir(maindir)
    outdir = "results/"
    print(strategy_names)
    for strategy in strategy_names:
        df = get_sim_results(maindir, strategy, model, outdir)
        print(df.head())
        print_sim_metrics(df, strategy)
