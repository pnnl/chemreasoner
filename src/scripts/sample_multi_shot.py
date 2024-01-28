"""A script to loop through single-shot queries."""
import json
import sys

from pathlib import Path

import pandas as pd

sys.path.append("src")
from datasets.reasoner_data_loader import get_state  # noqa:E402
from llm.azure_openai_interface import AzureOpenaiInterface  # noqa:E402

llm_function = AzureOpenaiInterface(".env")

df = pd.read_csv(Path("data", "input_data", "dataset.csv"))
num_samples = 10
for j in range(num_samples):
    data = []
    for i, row in df.iterrows():
        dataset = row["dataset"]
        query = row["query"]
        s = get_state(dataset, query, chain_of_thought=True)
        print(s.generation_prompt)

        system_prompt = s.generation_system_prompt
        prompt = s.generation_prompt

        answer = run_azure_openai_prompts([prompt], [system_prompt])
        s.process_generation(answer[0])
        with open(
            Path("multi_shot_results", f"sample_{j}", f"query_{str(i).zfill(3)}.json", "w")
        ) as f:
            json.dump(vars(s), f)
        data.append(vars(s))
    print(data)