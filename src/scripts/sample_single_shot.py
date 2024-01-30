"""A script to loop through single-shot queries."""
import json
import sys

from pathlib import Path

import pandas as pd

sys.path.append("src")
from datasets.reasoner_data_loader import get_state  # noqa:E402
from llm.llama2_vllm_chemreasoner import LlamaLLM  # noqa:E402

llm_func = LlamaLLM(model="meta-llama/Llama-2-13b-chat-hf", num_gpus=1)

df = pd.read_csv(Path("data", "input_data", "dataset.csv"))
data = []
for i, row in df.iterrows():
    dataset = row["dataset"]
    query = row["query"]
    s = get_state(dataset, query, chain_of_thought=True)
    print(s.generation_prompt)

    system_prompt = s.generation_system_prompt
    prompt = s.generation_prompt

    answer = llm_func([prompt], [system_prompt])
    s.process_generation(answer[0])
    Path("single_shot_result_llama2s").mkdir(parents=True, exist_ok=True)
    with open(
        Path("single_shot_results", f"single_shot_{str(i).zfill(3)}.json", "w")
    ) as f:
        json.dump(vars(s), f)
    data.append(vars(s))
print(data)
