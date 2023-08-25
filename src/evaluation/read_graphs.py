import json
import networkx as nx
import numpy as np

graph = nx.read_gpickle(
    "/Users/spru445/alchemist/catalysis_data/mcr_catalysis_48.gpickle"
)
# print(graph)
max_idx = np.argmax(
    [graph.nodes(data=True)[i]["node_rewards"] for i in range(len(graph.nodes))]
)
# print(graph.nodes(data=False)[max_idx])
print(max_idx)
sp = nx.all_simple_paths(graph, 0, max_idx)

answers = []
prompts = []
for node in list(sp)[0]:
    answers.append(graph.nodes()[node]["answer"])
    prompts.append(graph.nodes()[node]["prompt"])
    print("------")
    print(graph.nodes()[node]["node_rewards"])
    print(graph.nodes()[node]["prompt"])
    print(graph.nodes()[node]["answer"])
    print("-------")

output = {"answers": answers, "prompts": prompts}
with open("./trajectory.json", "w") as f:
    json.dump(output, f)
