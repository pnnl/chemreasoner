"""Run the queries for ICML."""

import time

start = time.time()
import argparse
import configparser
import json
import logging
import os
import sys

from pathlib import Path
from traceback import format_exc

import numpy as np
import pandas as pd

sys.path.append("src")
from datasets import reasoner_data_loader  # noqa:E402
from llm.azure_open_ai_interface import AzureOpenaiInterface  # noqa:E402
from llm.llama2_vllm_chemreasoner import LlamaLLM  # noqa:E402
from search.policy import coherent_policy, reasoner_policy  # noqa:E402
from search.reward import (
    simulation_reward,
    llm_reward,
    microstructure_search_reward,
)  # noqa:E402
from search.methods.tree_search.beam_search import BeamSearchTree  # noqa:E402
from search.state.reasoner_state import ReasonerState  # noqa:E402

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


def get_search_method(config, data, policy, reward_fn):
    """Get the search method provided in args."""
    if config.get("SEARCH", "search-method") == "beam-search":
        num_keep = config.getint("BEAM SEARCH", "num-keep")
        num_generate = config.getint("BEAM SEARCH", "num-generate")
        assert num_keep > 0, "invalid parameter"
        assert num_generate > 0, "invalid parameter"
        return BeamSearchTree(
            data,
            policy,
            reward_fn,
            num_generate=num_generate,
            num_keep=num_keep,
        )
    elif config.get("SEARCH", "search-method") == "mcts":
        raise NotImplementedError("Monte Carlo Tree Search is not implemented, yet.")
    else:
        raise NotImplementedError(
            f"Unkown Search strategy {config.get('SEARCH', 'search-method')}."
        )


def get_reward_function(config, state, llm_function):
    """Get the reward function provided in args."""
    penalty_value = config.getfloat("REWARD", "reward-function")
    assert penalty_value < 0, "invalid parameter"
    reward_max_attempts = config.getint("REWARD", "reward-max-attempts")
    assert reward_max_attempts > 0, "invalid parameter"

    # TODO: make these args follow config
    if config.get("REWARD", "reward_function") == "simulation-reward":
        assert (
            isinstance(args.nnp_class, str) and args.nnp_class == "oc"
        ), "invalid parameter"
        assert (
            isinstance(args.num_slab_samples, int) and args.num_slab_samples > 0
        ), "invalid parameter"
        assert (
            isinstance(args.num_adslab_samples, int) and args.num_adslab_samples > 0
        ), "invalid parameter"

        # check nnp_kwargs
        assert (
            isinstance(args.reward_max_attempts, int) and args.reward_max_attempts > 0
        ), "invalid parameter"
        assert args.gnn_model in [
            "gemnet-t",
            "gemnet-oc",
            "escn",
            "eq2",
        ], "invalid parameter"
        assert isinstance(args.gnn_traj_dir, str), "invalid parameter"
        assert (
            isinstance(args.gnn_batch_size, int) and args.gnn_batch_size > 0
        ), "invalid parameter"
        assert isinstance(args.gnn_device, str) and (
            args.gnn_device == "cpu" or args.gnn_device == "cuda"
        ), "invalid parameter"
        assert (
            isinstance(args.gnn_ads_tag, int) and args.gnn_ads_tag == 2
        ), "invalid parameter"
        assert (
            isinstance(args.gnn_fmax, float) and args.gnn_fmax > 0
        ), "invalid parameter"
        assert (
            isinstance(args.gnn_steps, int) and args.gnn_steps >= 0
        ), "invalid parameter"
        nnp_kwargs = {
            "model": args.gnn_model,
            "traj_dir": Path(args.gnn_traj_dir),
            "batch_size": args.gnn_batch_size,
            "device": args.gnn_device,
            "ads_tag": args.gnn_ads_tag,
            "fmax": args.gnn_fmax,
            "steps": args.gnn_steps,
        }
        return simulation_reward.StructureReward(
            llm_function=llm_function,
            penalty_value=args.penalty_value,
            nnp_class=args.nnp_class,
            num_slab_samples=args.num_slab_samples,
            num_adslab_samples=args.num_adslab_samples,
            max_attempts=args.reward_max_attempts,
            gnn_service_port=args.gnn_port,
            flip_negative=args.flip_negative,
            **nnp_kwargs,
        )
    elif config.get("REWARD", "reward_function") == "microstructure-reward":
        return microstructure_search_reward.StructureReward(
            llm_function=llm_function,
            microstructure_results_dir=Path(args.microstructure_results_dir),
            config=config,
        )

    elif config.get("REWARD", "reward_function") == "llm-reward":
        assert isinstance(args.reward_limit, float), "invalid parameter"
        return llm_reward.LLMRewardFunction(
            llm_function,
            reward_limit=args.reward_limit,
            max_attempts=args.reward_max_attempts,
            penalty_value=args.penalty_value,
        )
    else:
        raise NotImplementedError(f"Unknown reward function {args.reward_function}.")


def get_policy(config, llm_function: callable = None):
    """Get the policy provided in args."""
    if config.get("POLICY", "policy") == "coherent-policy":
        max_num_actions = config.getint("COHERENT POLICY", "max-num-actions")
        assert max_num_actions > 0
        policy_max_attempts = config.getint("COHERENT POLICY", "policy-max-attempts")
        assert policy_max_attempts > 0
        assert llm_function is not None
        return coherent_policy.CoherentPolicy(
            llm_function, max_num_actions, max_attempts=policy_max_attempts
        )
    elif config.get("POLICY", "policy") == "reasoner-policy":
        return reasoner_policy.ReasonerPolicy(try_oxides=False)


def get_state_from_idx(idx, df: pd.DataFrame):
    """Get the state referenced by idx."""
    dataset = df.iloc[idx]["dataset"]
    query = df.iloc[idx]["query"]
    return reasoner_data_loader.get_state(dataset, query, chain_of_thought=True)


def get_llm_function(config):
    """Get the llm function specified by args."""
    assert isinstance(config.get("MACRO SEARCH", "dotenv-path"), str)
    assert isinstance(config.get("MACRO SEARCH", "llm"), str)
    if config.get("MACRO SEARCH", "llm") in ["gpt-4", "gpt-3.5-turbo"]:
        llm_function = AzureOpenaiInterface(
            config.get("MACRO SEARCH", "dotenv-path"),
            model=config.get("MACRO SEARCH", "llm"),
        )
    elif config.get("MACRO SEARCH", "llm") == "llama2-13b":
        llm_function = LlamaLLM(
            "meta-llama/Llama-2-13b-chat-hf",
            num_gpus=1,
        )
    else:
        raise ValueError(f"Unkown LLM {config.get('MACRO SEARCH', 'llm')}.")

    return llm_function


def get_indeces(config):
    """Get the state indeces provided in args."""
    start_query = config.getint("MACRO SEARCH", "start-query")
    assert args.start_query >= 0
    end_query = config.getint("MACRO SEARCH", "end-query")
    assert start_query > start_query
    return list(range(start_query, end_query))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default=None)

    args = parser.parse_args()

    config = configparser.ConfigParser()

    config.read(args.config_path)

    depth = config.getint("MACRO SEARCH", "depth")
    assert depth > 0

    start = time.time()
    save_dir = Path(config.get)
    save_dir.mkdir(parents=True, exist_ok=True)

    llm_function = get_llm_function(config)

    df = pd.read_csv(config.get("MACRO SEARCH", "dataset-path"))
    indeces = get_indeces(config)
    end = time.time()
    logging.info(f"TIMING: Initialization time: {end-start}")

    for i in indeces:
        continue_searching = True
        try:
            logging.info(
                f"=============TIMING: Processing query {i}/{len(indeces)}================"
            )
            start = time.time()
            fname = save_dir / f"search_tree_{i}.json"
            starting_state = get_state_from_idx(i, df)

            policy = get_policy(config, llm_function)
            reward_fn = get_reward_function(config, starting_state, llm_function)

            if Path(fname).exists() and os.stat(fname).st_size != 0:
                print(f"Loading a tree from {fname}")
                logging.info("=" * 20 + " " + str(i) + " " + "=" * 20)
                with open(fname, "r") as f:
                    tree_data = json.load(f)
                    search = BeamSearchTree.from_data(
                        tree_data,
                        policy,
                        reward_fn,
                        node_constructor=ReasonerState.from_dict,
                    )
                    num_keep = config.getint("BEAM SEARCH", "num-keep")
                    assert num_keep == search.num_keep, "mismatch parameter"

                    num_generate = config.getint("BEAM SEARCH", "num-generate")
                    assert num_generate == search.num_generate, "mismatch parameter"
            else:
                search = get_search_method(config, starting_state, policy, reward_fn)

            end = time.time()
            logging.info(f"TIMING: Time to set up query: {end-start}")

            start_time = time.time()
            timing_data = [start_time]
            continue_searching = True
            while len(search) < depth and continue_searching:
                start = time.time()

                data = search.step_return()
                end_time = time.time()
                timing_data.append(end_time - timing_data[-1])
                with open(fname, "w") as f:
                    data.update(
                        {"total_time": end_time - start_time, "step_times": timing_data}
                    )
                    json.dump(data, f, cls=NpEncoder)

                end = time.time()
                logging.info(f"TIMING: One search iteration: {end-start}")

                logging.info("=" * 20 + " " + str(i) + " " + "=" * 20)
        except Exception as err:
            logging.warning(f"Could not complete search with error: {err}")
            logging.warning(format_exc())
            continue_searching = False
