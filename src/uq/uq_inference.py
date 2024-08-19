"""Functions and objects for GBMregressor uq model."""

import numpy as np
import os
import pickle
import logging
import xgboost as xgb

import torch


class GBMRegressor:
    """
    Union approach for Gradient Boosting Machine uncertainty estimation
    from https://link.springer.com/article/10.1186/s13321-023-00753-5
    """

    def __init__(
        self, model_path="./", lower_alpha=0.1, upper_alpha=0.9, n_estimators=100
    ):
        """Initialize GBM regressors
        Args:
          model_path (str): Path to pickle file containing trained GBM regressor.
                         (default: :obj:`./`)
          lower_alpha (float): The alpha-quantile of the quantile loss function.
                               Values must be in the range (0.0, 1.0).
                               (default: :obj:`0.1`)
          upper_alpha (float): The alpha-quantile of the quantile loss function.
                               Values must be in the range (0.0, 1.0).
                               (default: :obj:`0.9`)
          n_estimators (int): The number of boosting stages to perform.
                              (default: :obj:`100`)
        """
        self.model_path = model_path
        self.alpha = np.array([lower_alpha, upper_alpha])
        self.n_estimators = n_estimators

    def update(self, embeddings, target):
        """Update GBM models after training epoch."""
        Xy = xgb.QuantileDMatrix(embeddings, target)
        Xy_test = xgb.QuantileDMatrix(embeddings, target, ref=Xy)

        self.booster = xgb.train(
            {
                "objective": "reg:quantileerror",
                "tree_method": "hist",
                "quantile_alpha": self.alpha,
                "learning_rate": 0.04,
                "max_depth": 5,
                "verbosity": 0,
                "disable_default_eval_metric": True,
            },
            Xy,
            num_boost_round=self.n_estimators,
        )

    def predict(self, embeddings):
        """Predict uncertainties for set of embeddings."""

        scores = self.booster.inplace_predict(embeddings).T
        return np.abs(scores[0] - scores[1]) / 2

    def _save(self):
        """Save GBM regressor parameters to file."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.booster, f)

    def _load(self):
        """Load trained GBM regressors from file."""
        if os.path.isfile(self.model_path):
            with open(self.model_path, "rb") as f:
                self.booster = pickle.load(f)
        else:
            logging.warning(
                f"No trained GBM regressor found in {self.model_path}. Call GBMRegressor.update to train a model."
            )


def get_per_sample_embeddings(output_embeddings, batch):
    """
    Given a dictionary comtaining model output per batch of the form:
    {"energy": E_t, "hidden_h":h, "hidden_m":m, 'edge_index':edge_index}

    generate, embeddings per model input:
    [embeddings_atomistic_graph1, embeddings_atomistic_graph2.....embeddings_atomistic_graphN]

    """
    data = output_embeddings
    # print(data)
    atom_emb = data["hidden_h"].detach().to("cpu")
    edge_emb = data["hidden_m"].detach().to("cpu")
    energies = data["energy"].detach().to("cpu")
    forces = data["forces"].detach().to("cpu")
    graph_embs = []
    for i in range(len(batch.ptr) - 1):
        idx_start = batch.ptr[i]
        idx_end = batch.ptr[i + 1]
        # print(i, idx_start, idx_end)
        graph_emb = atom_emb[idx_start:idx_end]
        # print(graph_emb.size())
        graph_emb = torch.mean(graph_emb, 0)
        # print(graph_emb.size())
        graph_embs.append(graph_emb)
    return np.array(graph_embs)
