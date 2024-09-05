#!/bin/bash
python src/scripts/put_adsorption_into_db.py

python src/scripts/run_icml_queries.py --savedir results \
    --dataset-path data/input_data/dataset.csv \
    --start-query 2 \
    --end-query 3 \
    --depth 5 \
    --policy coherent-policy \
    --max-num-actions 10 \
    --policy-max-attempts 3 \
    --reward-function simulation-reward \
    --penalty-value -10 \
    --nnp-class oc \
    --num-slab-samples 16 \
    --num-adslab-samples 16 \
    --reward-max-attempts 3 \
    --gnn-model gemnet-t \
    --gnn-traj-dir trajectories_e_tot \
    --gnn-batch-size 40 \
    --gnn-device cuda \
    --gnn-ads-tag 2 \
    --gnn-fmax 0.05 \
    --gnn-steps 64 \
    --search-method beam-search \
    --num-keep 6 \
    --num-generate 8\
    --dotenv-path .env \
    --llm gpt-4