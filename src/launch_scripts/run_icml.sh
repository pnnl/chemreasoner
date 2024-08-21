#!/bin/bash
python src/scripts/put_adsorption_into_db.py

python src/scripts/run_icml_queries.py --savedir <results directory> \
    --dataset-path data/input_data/dataset.csv \
    --start-query <index of starting query> \
    --end-query <index of ending query> \
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
    --gnn-traj-dir <path to save trajectories> \
    --gnn-batch-size 40 \
    --gnn-device cuda \
    --gnn-ads-tag 2 \
    --gnn-fmax 0.05 \
    --gnn-steps 64 \
    --search-method beam-search \
    --num-keep 6 \
    --num-generate 8\
    --dotenv-path <path to azure OpenAI .env file> \
    --llm gpt-4