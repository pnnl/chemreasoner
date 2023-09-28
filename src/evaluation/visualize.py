"""Functions to visualize a search tree."""
import pickle
import warnings

from pathlib import Path

import numpy as np

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

import matplotlib.pyplot as plt

import plotly.graph_objects as go


def unpickle_data(fname: Path):
    """Read in search tree data from a pickle file."""
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def remove_colons(tree_data):
    """Remove colons from tree data to prevent issues."""
    for i, n in enumerate(tree_data["nodes"]):
        new_n = dict()
        for k, v in n.items():
            new_k = k.replace(":", ";")
            if isinstance(v, str):
                new_v = v.replace(":", ";")
            else:
                new_v = v
            new_n[new_k] = new_v
        tree_data["nodes"][i] = new_n


def orient_attributes(keys: list, attr_dict: dict[list]):
    """Orient the edge attributes dictionary."""
    output = dict()
    for i, e in enumerate(keys):
        temp_dict = dict()
        for k, v in attr_dict.items():

            temp_dict[k] = v[i]
        output[e] = temp_dict
    return output


def search_tree_to_nx(tree_data) -> nx.Graph:
    """Turn a search tree into a networkx graph."""
    node_attr = dict()
    for k in tree_data["nodes"][0]:
        if not isinstance(tree_data["nodes"][0][k], list):
            node_attr[k] = [n[k] for n in tree_data["nodes"]]
        else:
            node_attr[k] = [", ".join(str(n[k])) for n in tree_data["nodes"]]

    node_attr["total_visits"] = np.sum(tree_data["children_visits"], axis=1)
    node_attr["node_rewards"] = tree_data["node_rewards"]
    node_attr["children_rewards"] = np.sum(tree_data["children_rewards"], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        node_attr["downstream_rewards"] = np.nan_to_num(
            node_attr["children_rewards"] / node_attr["total_visits"]
        )
    node_attr = orient_attributes(list(range(len(tree_data["nodes"]))), node_attr)
    child_idx = tree_data["children_idx"]
    edges = [
        (i, child_idx[i, j])
        for i in range(child_idx.shape[0])
        for j in range(child_idx.shape[1])
        if child_idx[i, j] != -1
    ]
    edge_attr = dict()
    for label in [
        "children_priors",
        "children_rewards",
        "children_visits",
    ]:
        arr = tree_data[label]

        edge_attr[label] = np.array(
            [
                arr[i, j]
                for i in range(child_idx.shape[0])
                for j in range(child_idx.shape[1])
                if child_idx[i, j] != -1
            ]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        edge_attr["Q"] = np.nan_to_num(
            edge_attr["children_rewards"] / edge_attr["children_visits"], posinf=0
        )

    corr_node_idx = np.array([int(e[0]) for e in edges])
    edge_attr["u"] = (
        edge_attr["children_priors"][corr_node_idx]
        * np.sqrt(np.sum(edge_attr["children_visits"]))
        / (1 + edge_attr["children_visits"])
    )
    edge_attr["action"] = edge_attr["Q"] + 3 * edge_attr["u"]
    edge_attr["log_action"] = np.log(edge_attr["action"])

    edge_attr = orient_attributes(edges, edge_attr)
    G = nx.Graph(edges)
    nx.set_node_attributes(G, node_attr)
    nx.set_edge_attributes(G, edge_attr)
    return G


def make_annotations(pos, text, labels, M, font_size=10, font_color="rgb(250,250,250)"):
    """Make annotations for the nodes in a figure."""
    L = len(pos)
    if len(text) != L:
        raise ValueError("The lists pos and text must have the same len")
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k],
                x=pos[k][0],
                y=2 * M - pos[k][1],
                xref="x1",
                yref="y1",
                font=dict(color=font_color, size=font_size),
                showarrow=False,
            )
        )
    return annotations


def nx_to_plotly_figure(G: nx.Graph) -> go.Figure:
    """Make figure from nx graph."""
    nr_vertices = len(G.nodes())
    v_label = list(range(len(G.nodes)))
    hover_v_label = [int(attr["total_visits"]) for _, attr in G.nodes(data=True)]
    G = Graph.from_networkx(G)
    lay = G.layout("rt")

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    # es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    labels = v_label

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            name="bla",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color="#6175c1",  # '#DB4551',
                line=dict(color="rgb(50,50,50)", width=1),
            ),
            text=labels,
            hoverinfo="text",
            hovertext=hover_v_label,
            opacity=0.8,
        )
    )
    axis = dict(
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )
    fig.update_layout(
        title="Tree with Reingold-Tilford Layout",
        annotations=make_annotations(position, v_label, labels, M),
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
    )
    return fig


def nx_plot(H: nx.Graph, node_attr="node_rewards", edge_attr="log_action", title=None):
    """Visualize the graph using networkx vizualizations."""
    G = nx.Graph()
    G.add_nodes_from(sorted(H.nodes(data=True)))
    G.add_edges_from(H.edges(data=True))
    fig, ax = plt.subplots(1, 1)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    node_colors = np.log(np.array(list(nx.get_node_attributes(G, node_attr).values())))
    node_cmap = plt.cm.viridis
    node_vmin = np.partition(np.sort(node_colors), 2)[1]  # Second lowest value
    node_vmax = max(node_colors)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=12,
        node_color=node_colors,
        cmap=node_cmap,
        vmin=node_vmin,
        vmax=node_vmax,
        label={i: i for i in range(len(G.nodes()))},
        # font_color="w",
    )
    node_sm = plt.cm.ScalarMappable(
        cmap=node_cmap, norm=plt.Normalize(vmin=node_vmin, vmax=node_vmax)
    )
    node_sm._A = []
    # plt.colorbar(node_sm, label="Node Reward")

    # Plot Edges
    edge_colors = list(nx.get_edge_attributes(G, edge_attr).values())
    edge_cmap = plt.cm.viridis
    edge_vmin = min(edge_colors)
    edge_vmax = max(edge_colors)
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
    )

    edge_sm = plt.cm.ScalarMappable(
        cmap=edge_cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    )
    edge_sm._A = []
    # plt.colorbar(edge_sm, label="Action Value")
    ax = plt.gca()
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind):
        node = ind["ind"][0]
        xy = pos[node]
        annot.xy = xy
        node_attr = {"node": node}
        node_attr.update(G.nodes[node])
        text = "\n".join(
            f"{k}: {v}"
            for k, v in node_attr.items()
            if k in ["node_rewards", "node_idx"]
        )
        annot.set_text(text)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = nodes.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if title is not None:
        ax.set_title(title)


if __name__ == "__main__":
    tree_data = unpickle_data("catalysis_data/mcr_catalysis_35.pkl")
    print(tree_data["nodes"])
    for i, n in enumerate(tree_data["nodes"]):
        new_n = dict()
        for k, v in n.items():
            new_k = k.replace(":", ";")
            if isinstance(v, str):
                new_v = v.replace(":", ";")
            else:
                new_v = v
            new_n[new_k] = new_v
        tree_data["nodes"][i] = new_n
    print(f"Elapsed time: {tree_data['end_time'] - tree_data['start_time']}")
    # for k, v in tree_data.items():
    #     if k not in ["tradeoff", "discount"]:
    #         print(k, len(v))

    G = search_tree_to_nx(tree_data)
    nx_plot(G)
