import argparse
import requests
import sys

from pathlib import Path

from flask import Flask, request, render_template, jsonify, Response

sys.path.append("src")
from search.reward import simulation_reward  # noqa: E402

app = Flask(__name__)

parser = argparse.ArgumentParser()
# Reward function
parser.add_argument("--gnn-port", required=True, help="Specify server port")
parser.add_argument("--gnn-traj-dir", type=str, default=None)

args = parser.parse_args()


def get_structure_reward(args):
    """Argument parser parse arguments."""

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


struct_reward_provider = get_structure_reward(args)


def actual_gnn_func(request):
    slab_syms = request.json.get("slab_syms")
    ads_list = request.json.get("ads_list")
    candidates_list = request.json.get("candidates_list")
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
    output_dict = actual_gnn_func(request)
    response = jsonify(output_dict)
    return response


if __name__ == "__main__":
    print("creating flask server")
    app.run(host="0.0.0.0", port=int(args.gnn_port))
