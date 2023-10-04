"""Create the scatterplot comparing adsorption energy methods (llm vs simulation)."""
import json

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_json(Path("data", "output", "scatterplot_data.json"))
df = df.drop_duplicates(subset=["adslab"])
names = np.array(df["adslab"])
print(names)

fig, ax = plt.subplots()

sc = plt.scatter(df["llm_value"], -df["simulation_value"], alpha=0.5)
plt.plot([0, 2.5], [0, 2.5])
plt.title("Conparison of GNN vs. LLM adsorption energies.")
plt.xlabel(r"$|\Delta E_{GPT}|$")
plt.ylabel(r"$- \Delta E_{sim}$")

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

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos

    text = f"{' '.join([names[n] for n in ind['ind']])}"
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

plt.show()
