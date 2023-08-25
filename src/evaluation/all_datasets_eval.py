"""Evaluate the results of all datasets."""
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("src")
from evaluation.oc_evaluation import parse_oc_data
from evaluation.catalysis_evaluation import parse_catalysis_data


def collapse_multi_shot(df, agg_mode="mean"):
    """Collapse multi shot rows by the given aggregation function."""
    df_filtered = df.loc[["multi_shot_" in i for i in df.index], :]
    if agg_mode == "mean":
        final_series = df_filtered.mean(axis=0)
        return final_series
    if agg_mode == "max":
        final_series = df_filtered.max(axis=0)
        return final_series
    if agg_mode == "sum":
        final_series = df_filtered.sum(axis=0)
        return final_series
    else:
        raise ValueError(f"Aggregation mode {agg_mode} not supported.")


if __name__ == "__main__":
    # OC dataset
    oc_reward, oc_calls = parse_oc_data("post_submission_tests_davinci/oc")
    print(oc_reward)

    oc_reward.loc["multi_shot"] = collapse_multi_shot(oc_reward, "max")
    oc_calls.loc["multi_shot"] = collapse_multi_shot(oc_calls, "sum")
    print("oc")
    print(oc_reward.mean(axis=1))
    # Catalysis experts dataset
    catalysis_reward, catalysis_calls = parse_catalysis_data(
        "post_submission_tests_davinci/biofuel"
    )
    catalysis_reward.loc["multi_shot"] = collapse_multi_shot(catalysis_reward, "max")
    catalysis_calls.loc["multi_shot"] = collapse_multi_shot(catalysis_calls, "sum")
    print("biofuels")
    print(catalysis_reward.mean(axis=1))
