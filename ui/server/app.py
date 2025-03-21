import json
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from io import StringIO
from os import listdir
from os.path import dirname, join, realpath
from typing import Any, Dict, List

from dotenv import load_dotenv  # type: ignore
from flask import Flask, jsonify, send_file  # type: ignore
from flask_compress import Compress  # type: ignore

load_dotenv()

rootdir = dirname(realpath(__file__))
datadir = join(rootdir, "data")
distdir = join(rootdir, "..", "client", "dist")

SEARCH_TREE = json.load(open(join(datadir, 'search_tree_6.json')))

NODE_LOOKUP = {}


def traverse_tree(node, parent_id=None):
    node_id = str(node["id"])
    NODE_LOOKUP[node_id] = node
    node["parent_id"] = parent_id
    for child in node.get("children", []):
        traverse_tree(child, node_id)


traverse_tree(SEARCH_TREE)


# ---------------------------------------------------------------------------------------


app = Flask(
    __name__,
    static_url_path="",
    static_folder=distdir,
)
Compress(app)
app.config["COMPRESS_MIMETYPES"] = [
    "application/javascript",
    "application/json",
    "application/xyz",
    "text/css",
    "text/html",
    "text/javascript",
    "text/xml",
]


@app.route("/")
def home():
    return send_file(f"{distdir}/index.html")


@app.route("/api/search-tree")
def get_graph():
    return jsonify(SEARCH_TREE)


@app.route("/api/structures/<node_id>")
def get_structures(node_id):
    node = NODE_LOOKUP.get(node_id)
    if node is None:
        return jsonify({"error": f"Node {node_id} not found"}), 404
    reaction_pathways = node.get("reaction_pathways", [])
    catalysts = node.get("info", {}).get("symbols", [{}])[0].get("symbols")
    if not reaction_pathways or not catalysts:
        return jsonify({"error": "No reaction pathways or catalysts found"}), 404

    def get_xyz_energy(xyz: str) -> float:
        for line in StringIO(xyz).readlines():
            # Assuming the line format is something like: "...energy=-123.456..."
            m = re.search(r"energy=(-?\d+(?:\.\d+)?)", line)
            if m:
                return float(m.group(1))
        return 0.0

    def get_structure_data(path_prefix: str) -> dict | None:
        print(f"PREFIX: {path_prefix}")
        try:
            names = listdir(join(datadir, path_prefix))
            paths = [join(datadir, path_prefix, name) for name in names]
        except OSError:
            # Since these paths have an asterisk embedded in them, Windows will throw an
            # error but glob might work...
            paths = glob(join(datadir, path_prefix, "*.xyz"))
        if not paths:
            return None
        structure = open(paths[0]).read()
        return {
            "structure": structure,
            "energy": get_xyz_energy(structure),
            # "energy": read_xyz(StringIO(structure)).get_potential_energy(),
        }

    def get_structure_prefix(reactant: str, catalyst: list[str]) -> str:
        catalyst_str = "".join([catalyst[0]] + sorted(catalyst[1:]))
        return join("processed_structures", f"{catalyst_str}_{reactant}", "")

    def fetch_structures(
        reactants: List[str], catalysts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_prefix = {
                executor.submit(
                    get_structure_data, get_structure_prefix(reactant, catalyst)
                ): get_structure_prefix(reactant, catalyst)
                for catalyst in catalysts
                for reactant in reactants
            }
            results = {
                future_to_prefix[future]: future.result()
                for future in as_completed(future_to_prefix)
            }
            ordered_results = [
                results[get_structure_prefix(reactant, catalyst)]
                for catalyst in catalysts
                for reactant in reactants
            ]
            return [
                ordered_results[i : i + len(reactants)]  # noqa: E203
                for i in range(0, len(ordered_results), len(reactants))
            ]

    pathways_data = [
        {
            "reactants": reactants,
            "structures": fetch_structures(reactants, catalysts),
        }
        for reactants in reaction_pathways
    ]

    return jsonify(
        {
            "catalysts": catalysts,
            "pathways": pathways_data,
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", default=8000, type=int)
    parser.add_argument("-e", "--expose", action="store_true")
    args = parser.parse_args()
    host = "127.0.0.1"
    if args.expose:
        host = "0.0.0.0"
    app.run(debug=True, host=host, port=args.port)
