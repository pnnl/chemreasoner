import argparse
import requests
import time
from flask import Flask, request, render_template, jsonify

def simulate_gnn_call(catalyst_arg, server_port):
    params = {"catalyst_id": catalyst_arg}
    url = f'http://localhost:{server_port}/GemNet'
    response = requests.post(url, params)

    if response.status_code == 200:
        # Process the response data
        print('Request successful!')
        print(response.json())
    else:
        print(f"Request failed with status code {response.status_code}")
    return response.json()

def run_test(server_port):
    port = int(server_port)
    results = []
    start = time.time()
    arg = "Cu-Zn-Ni"
    results.append(simulate_gnn_call(arg, port))
    end = time.time()
    print(f'Total time: {end-start}')
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', required=True, help="Specify server port")
    args = parser.parse_args()
    run_test(args.server_port)
