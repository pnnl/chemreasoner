#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import os
import pickle
import logging
import xgboost as xgb

import oc20
import sys
import torch
from pathlib import Path
from torch_geometric.data import Batch
from torch_geometric.loader.data_list_loader import DataListLoader


sys.path.append("/people/d3x771/projects/chemreasoner/chemreasoner/src")
from nnp.oc import OCAdsorptionCalculator
 
class GBMRegressor:
    """
    Union approach for Gradient Boosting Machine uncertainty estimation
    from https://link.springer.com/article/10.1186/s13321-023-00753-5 
    """
    def __init__(self, savedir='./', lower_alpha=0.1, upper_alpha=0.9, n_estimators=100):
        """Initialize GBM regressors
        Args:
          savedir (str): Directory to save fit GBM regressors. 
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
        self.savedir = savedir
        self.alpha = np.array([lower_alpha, upper_alpha])
        self.n_estimators = n_estimators
        
    @property
    def model_file(self):
        return 'GBMRegressor.pkl'
        
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
        return np.abs(scores[0]-scores[1])/2
 
    def _save(self):
        """Save GBM regressor parameters to file."""
        with open(os.path.join(self.savedir, self.model_file), 'wb') as f:
            pickle.dump(self.booster, f)
 
 
    def _load(self):
        """Load trained GBM regressors from file."""
        if os.path.isfile(os.path.join(self.savedir, self.model_file)):
            with open(os.path.join(self.savedir, self.model_file), 'rb') as f:
                self.booster = pickle.load(f)
        else:
            logging.warning(f'No trained GBM regressor found in {self.savedir}. Call GBMRegressor.update to train a model.')


# In[5]:


def get_per_sample_embeddings(output_embeddings, batch):
    """
    Given a dictionary comtaining model output per batch of the form:
    {"energy": E_t, "hidden_h":h, "hidden_m":m, 'edge_index':edge_index}
    
    generate, embeddings per model input:
    [embeddings_atomistic_graph1, embeddings_atomistic_graph2.....embeddings_atomistic_graphN]

    """
    data = output_embeddings
    #print(data)
    atom_emb = data['hidden_h']
    edge_emb = data['hidden_m']
    energies = data['energy']
    forces = data['forces']
    graph_embs = []
    for i in range(len(batch.ptr)-1):
        idx_start = batch.ptr[i]
        idx_end = batch.ptr[i+1]
        #print(i, idx_start, idx_end)
        graph_emb = atom_emb[idx_start:idx_end]
        #print(graph_emb.size())
        graph_emb = torch.mean(graph_emb, 0)
        #print(graph_emb.size())
        graph_embs.append(graph_emb)
    return(np.array(graph_embs))


# In[6]:


def init_model(model_type, batch_size):
    if(model_type == "gemnet-t"):
        ads_calc = OCAdsorptionCalculator(
            **{
                "model": "gemnet-t",  # change to gemnet-oc-22 TODO: 06/10/2024
                "traj_dir": Path("irrelevant"),
                "batch_size": 40,
                "device": "cpu",
                "ads_tag": 2,
                "fmax": 0.05,
                "steps": 300,
            }
        )
    else:
        ads_calc = OCAdsorptionCalculator(
            **{

                "model": "gemnet-oc-22",
                "traj_dir": Path("irrelevant"),
                "batch_size": batch_size,
                "device": "cuda",
                "ads_tag": 2,
                "fmax": 0.03,
                "steps": 200,
            }
        )      

    torch_calc = ads_calc.get_torch_model
    return torch_calc


# In[7]:




if(dataset == dataset_type):
        datadir= '/qfs/projects/chemreasoner/data/OC20/'
        batch_size = 32
        dataset = oc20.OC20(datadir, tag='200k')
        loader = DataListLoader(dataset, batch_size=batch_size, shuffle=False)
else:
        print("UNSUPPORTED DATASET FORMAT, exiting...")
        sys.exit(1)


    
torch_calc = init_model(model_type="gemnet-oc-22", batch_size=32)

uq_model = GBMRegressor(savedir="/people/d3x771/projects/chemreasoner/chemreasoner/")
uq_model._load()


def get_uq(torch_calc, uq_model):
    outputs = torch_calc.predict(batch,per_image=False)
    batch_embeddings = get_per_sample_embeddings(torch_calc.model.model_outemb, batch)
    batch_uq = uq_model.predict(batch_embeddings)
    print(batch_uq)
    return batch_uq


# In[ ]:


#print(uq)


# In[ ]:


#print(len(uq))


# In[ ]:





# In[ ]:




