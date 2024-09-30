import concurrent.futures
from io import StringIO
import json
import logging
import os
import re
from typing import List, Dict, Any

# from ase.io.xyz import read_xyz
import azure.functions as func
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
)
from azure.core.exceptions import ResourceNotFoundError

from node_context import NodeContext
from azure_openai_handler import AzureOpenAIHandler
from micro_structure import MicroStructureAgent
from llm_log_agent import LLMLogAgent


def get_storage_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(
        os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    )


def get_container_client() -> ContainerClient:
    return get_storage_client().get_container_client(os.environ["AZURE_CONTAINER_NAME"])


def get_blob_client(blob_name: str) -> BlobClient:
    return get_storage_client().get_blob_client(
        container=os.environ["AZURE_CONTAINER_NAME"], blob=blob_name
    )


def get_blob_lines(blob_name: str) -> list[str]:
    stream = get_blob_client(blob_name).download_blob()
    return stream.readall().decode("utf-8").splitlines()


def list_blobs(prefix: str) -> list[str]:
    return [
        blob.name for blob in get_container_client().list_blobs(name_starts_with=prefix)
    ]


def from_storage(req: func.HttpRequest, blob_name: str) -> func.HttpResponse:
    try:
        download_stream = get_blob_client(blob_name).download_blob()
        content_type = download_stream.properties.content_settings.content_type
        return func.HttpResponse(
            body=download_stream.readall(), mimetype=content_type, status_code=200
        )
    except ResourceNotFoundError:
        return func.HttpResponse(
            f"The blob {blob_name} was not found.", status_code=404
        )
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)


# ---------------------------------------------------------------------------------------


SEARCH_TREE = json.loads(
    get_blob_client("search_tree_6.json").download_blob().readall().decode("utf-8")
)
REWARDS = StringIO(
    get_blob_client("cu_zn_co_to_methanol/reward_values.csv")
    .download_blob()
    .readall()
    .decode("utf-8")
)

NODE_LOOKUP = {}


def traverse_tree(node, parent_id=None):
    node_id = str(node["id"])
    NODE_LOOKUP[node_id] = node
    node["parent_id"] = parent_id
    for child in node.get("children", []):
        traverse_tree(child, node_id)


traverse_tree(SEARCH_TREE)

GLOBAL_NODE_CONTEXT = NodeContext(SEARCH_TREE)
AZURE_HANDLER = AzureOpenAIHandler(".env")
MICRO_AGENT = MicroStructureAgent(REWARDS, AZURE_HANDLER)
# AGENT = LLMLogAgent(NODE_CONTEXT, AZURE_HANDLER, MICRO_AGENT)

# ---------------------------------------------------------------------------------------

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="search-tree", methods=["GET"])
def get_search_tree(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(SEARCH_TREE), mimetype="application/json")


@app.route(route="structures/{node_id}", methods=["GET"])
def get_structures(req: func.HttpRequest) -> func.HttpResponse:
    node_id = req.route_params.get("node_id")
    node = NODE_LOOKUP.get(node_id)
    if node is None:
        return func.HttpResponse(f"Node {node_id} not found", status_code=404)
    reaction_pathways = node.get("reaction_pathways", [])
    catalysts = node.get("info", {}).get("symbols", [{}])[0].get("symbols")
    if not reaction_pathways or not catalysts:
        return func.HttpResponse("No structures found", status_code=404)

    def get_xyz_energy(xyz: str) -> float:
        for line in StringIO(xyz).readlines():
            # Assuming the line format is something like: "...energy=-123.456..."
            m = re.search(r"energy=(-?\d+(?:\.\d+)?)", line)
            if m:
                return float(m.group(1))
        return 0.0

    def get_structure_data(path_prefix: str) -> dict | None:
        blob_names = list_blobs(path_prefix)
        if not blob_names:
            return None
        structure = (
            get_blob_client(blob_names[0]).download_blob().readall().decode("utf-8")
        )
        return {
            "structure": structure,
            "energy": get_xyz_energy(structure),
            # "energy": read_xyz(StringIO(structure)).get_potential_energy(),
        }

    def get_structure_prefix(reactant: str, catalyst: list[str]) -> str:
        catalyst_str = "".join([catalyst[0]] + sorted(catalyst[1:]))
        return f"node_structures/processed_structures/{catalyst_str}_{reactant}/"

    def fetch_structures(
        reactants: List[str], catalysts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    get_structure_data, get_structure_prefix(reactant, catalyst)
                )
                for catalyst in catalysts
                for reactant in reactants
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
            return [
                results[i : i + len(reactants)]  # noqa: E203
                for i in range(0, len(results), len(reactants))
            ]

    pathways_data = [
        {
            "reactants": reactants,
            "structures": fetch_structures(reactants, catalysts),
        }
        for reactants in reaction_pathways
    ]

    return func.HttpResponse(
        json.dumps(
            {
                "catalysts": catalysts,
                "pathways": pathways_data,
            }
        ),
        mimetype="application/json",
    )


@app.route(route="prompt/{node_id}", methods=["POST"])
def post_prompt(req: func.HttpRequest) -> func.HttpResponse:
    node_id = req.route_params.get("node_id")
    node = NODE_LOOKUP.get(node_id)
    if not node:
        return func.HttpResponse(f"Node {node_id} not found", status_code=404)
    body = req.get_json()
    prompt = body.get("prompt")
    if not prompt:
        return func.HttpResponse("No prompt provided", status_code=400)
    node_ids = [node["id"]]
    while node["parent_id"]:
        parent = NODE_LOOKUP[node["parent_id"]]
        node_ids.append(parent["id"])
        node = parent
    context = GLOBAL_NODE_CONTEXT.get_catalyst_recommendation_context(
        reversed(node_ids)
    )
    logging.info(context)
    agent = LLMLogAgent(GLOBAL_NODE_CONTEXT, AZURE_HANDLER, MICRO_AGENT)
    response = agent.process_query(prompt, context)
    return func.HttpResponse(
        json.dumps(
            {
                "response": response,
                "context": context,
            }
        ),
        mimetype="application/json",
    )
