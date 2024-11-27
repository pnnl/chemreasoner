"""Rin the microstructure search with the given config file."""

"""Run the queries for ICML."""

import time

start = time.time()
import argparse
import configparser
import json
import logging
import shutil
import sys

from pathlib import Path

import numpy as np

sys.path.append("src")
from search.methods.tree_search.microstructure_tree_search import (
    run_microstructure_search,
)
from llm.azure_open_ai_interface import AzureOpenaiInterface


end = time.time()

logging.getLogger().setLevel(logging.INFO)

logging.info(f"TIMING: Imports finished {end-start}")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super(NpEncoder, self).default(obj)


def get_llm_function(config):
    """Get the llm function specified by args."""
    assert isinstance(config.get("DEFAULT", "dotenv-path"), str)
    assert isinstance(config.get("DEFAULT", "llm"), str)
    if "gpt" in config.get("DEFAULT", "llm") or "o1-preview" in config.get(
        "DEFAULT", "llm"
    ):
        llm_function = AzureOpenaiInterface(
            config.get("DEFAULT", "dotenv-path"),
            model=config.get("DEFAULT", "llm"),
        )
    elif config.get("DEFAULT", "llm") == "llama2-13b":
        from llm.llama2_vllm_chemreasoner import LlamaLLM  # noqa:E402

        llm_function = LlamaLLM(
            "meta-llama/Llama-2-13b-chat-hf",
            num_gpus=1,
        )
    else:
        raise ValueError(f"Unkown LLM {config.get('DEFAULT', 'llm')}.")

    return llm_function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--catalyst-symbols", type=list_of_strings, default=None)

    args = parser.parse_args()

    config = configparser.ConfigParser()

    config.read(args.config_path)

    start = time.time()
    save_dir = Path(config.get("DEFAULT", "savedir"))
    save_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config_path, Path(save_dir) / "config.ini")

    llm_function = get_llm_function(config)

    if args.catalyst_symbols is None:
        catalyst_symbols = json.loads(
            config.get("MICROSTRUCTURE SEARCH", "catalyst-symbols")
        )
    else:
        catalyst_symbols = args.catalyst_symbols

    run_microstructure_search(
        config=config,
        catalyst_symbols=catalyst_symbols,
        save_dir=save_dir,
        llm_function=llm_function,
    )
