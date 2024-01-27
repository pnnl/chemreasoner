import argparse
import time
import requests
from flask import Flask, request, render_template, jsonify, Response

app = Flask(__name__)

# TODO - initialize the StructureReward
struct_reward_provider = StructureReward(TODO)

def actual_gnn_func():
    slab_syms = request.values.get("slab_syms")
    ads_list = request.values.get("ads_list")
    candidates_list = request.values.get("candidates_list")

    adslabs_and_energies,
                    gnn_calls,
                    gnn_time,
                    name_candidate_mapping = struct_reward_provider.create_structures_and_calculate(slab_syms,
                ads_list, candidates_list)
    output_dict = {
        "adslabs_and_energies": adslabs_and_energies,
        "gnn_calls": gnn_calls,
        "gnn_time": gnn_time,
        "name_candidate_mapping": name_candidate_mapping
    }
    response = jsonify(output_dict)
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
    print(f"catalyst_id: {input}")
    time.sleep(5)
    output_dict = {
            "energy": -1,
        }

    #
    # Convert output data structure to JSON response and return
    #
    response = jsonify(output_dict)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help="Specify server port")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=int(args.port))
