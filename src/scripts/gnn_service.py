import argparse
import requests
import sys
import time

from pathlib import Path

from flask import Flask, request, render_template, jsonify, Response

sys.path.append("src")
from search.reward import simulation_reward  # noqa: E402

app = Flask(__name__)

parser = argparse.ArgumentParser()
# Reward function
parser.add_argument("--port", required=True, help="Specify server port")
parser.add_argument("--gnn-traj-dir", type=str, default=None)

args = parser.parse_args()


def get_structure_reward(args):
    """Argument parser parse arguments."""
    if args.reward_function == "simulation-reward":
        assert isinstance(args.gnn_traj_dir, str), "invalid parameter"
        nnp_kwargs = {
            "model": "gemnet-t",
            "traj_dir": Path(args.gnn_traj_dir),
            "batch_size": 40,
            "device": "cuda",
            "ads_tag": 2,
            "fmax": 0.05,
            "steps": 64,
        }
        return simulation_reward.StructureReward(
            llm_function=None,
            penalty_value=-10,
            nnp_class="oc",
            num_slab_samples=16,
            num_adslab_samples=16,
            max_attempts=3,
            **nnp_kwargs,
        )


# TODO - initialize the StructureReward
struct_reward_provider = get_structure_reward(args)


def actual_gnn_func():
    slab_syms = request.values.get("slab_syms")
    ads_list = request.values.get("ads_list")
    candidates_list = request.values.get("candidates_list")
    (
        adslabs_and_energies,
        gnn_calls,
        gnn_time,
        name_candidate_mapping,
    ) = struct_reward_provider.create_structures_and_calculate(
        slab_syms, ads_list, candidates_list
    )
    response = {
        "adslabs_and_energies": adslabs_and_energies,
        "gnn_calls": gnn_calls,
        "gnn_time": gnn_time,
        "name_candidate_mapping": name_candidate_mapping,
    }
    return response


@app.route("/GemNet", methods=["POST"])
def GemNet():
    #
    # Use the get(..) semantics to extract key-value pairs
    #
    input = request.values.get("catalyst_id")

    #
    # Doing something random here
    #
    output_dict = actual_gnn_func(input)

    #
    # Convert output data structure to JSON response and return
    #
    response = jsonify(output_dict)
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(args.port))
