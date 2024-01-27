import argparse
import time
import requests
from flask import Flask, request, render_template, jsonify, Response

app = Flask(__name__)

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
