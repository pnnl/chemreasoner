#!/bin/bash

results_dir="/anfhome/rounak.meyur/chemreasoner_results"
traj_dir="/anfhome/rounak.meyur/chemreasoner_trajectories"
env_path="/anfhome/rounak.meyur/chemreasoner/.env"
start_query_index=3
end_query_index=7

python src/scripts/put_adsorption_into_db.py

python src/scripts/run_icml_queries.py --savedir $results_dir \
    --dataset-path data/input_data/dataset.csv \
    --start-query $start_query_index \
    --end-query $end_query_index \
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
    --gnn-model gemnet-oc-22 \
    --gnn-traj-dir $traj_dir \
    --gnn-batch-size 40 \
    --gnn-device cuda \
    --gnn-ads-tag 2 \
    --gnn-fmax 0.05 \
    --gnn-steps 64 \
    --search-method beam-search \
    --num-keep 6 \
    --num-generate 8\
    --dotenv-path $env_path \
    --llm gpt-4