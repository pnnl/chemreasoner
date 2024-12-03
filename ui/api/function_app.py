import logging
import os

import azure.functions as func  # type: ignore
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
)  # type: ignore
from azure.core.exceptions import ResourceNotFoundError  # type: ignore


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

FINAL_PREFIX = os.getenv("FINAL_PREFIX")


@app.route(route="data")
def api(req: func.HttpRequest) -> func.HttpResponse:
    return from_storage(req, "search_tree_6.json")


@app.route(route="structures")
def structures(req: func.HttpRequest) -> func.HttpResponse:
    return from_storage(req, f"{FINAL_PREFIX}/structures_with_rewards.json")


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
