import sys
import time

import unittest

sys.path.append("src")
from llm.query import run_prompts  # noqa: E402
from search.reward.llm_reward import LLMRewardFunction  # noqa: E402
from search.state.reasoner_state import ReasonerState  # noqa: E402
from search.methods.tree_search.beam_search import BeamSearchTree  # noqa: E402
from search.policy.reasoner_policy import ReasonerPolicy  # noqa: E402

adsorbates = ["CO", "H2O"]
template = (
    "Generate a list of top-5 {catalyst_label} "
    f"for the adsorption of {' and '.join(adsorbates)}."
    "{include_statement}{exclude_statement}"
    "Provide scientific explanations for each of the catalysts. "
    "Finally, return a python list named final_answer which contains the top-5 catalysts. "
    "{candidate_list_statement}"
    r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
)
starting_state = ReasonerState(
    template=template,
    reward_template=None,
    ads_symbols=adsorbates,
    ads_preferences=[1, -1],
    num_answers=5,
    prediction_model="llama-2",
    reward_model="llama-2",
    debug=True,
)

policy = ReasonerPolicy(
    catalyst_label_types=["", "monometallic ", "bimetallic ", "trimetallic "],
    try_oxides=False,
)


def test_parallel_prompts(prompts, system_prompts):
    assert len(prompts) > 1, "should pass multiple prompts to llm"
    return ["final_answer =['Pt','Cu','Ag']" for p, s in zip(prompts, system_prompts)]


def open_ai_sequential_prompts(prompts, system_prompts):
    answers = []
    for prompt, system_prompt in zip(prompts, system_prompts):
        answers += run_prompts([prompt], [system_prompt])
    return answers


gpt_reward_function = LLMRewardFunction(
    llm_function=run_prompts,
)

sequential_reward_function = LLMRewardFunction(
    llm_function=open_ai_sequential_prompts,
)

test_reward_function = LLMRewardFunction(
    llm_function=test_parallel_prompts,
)


class TestBeamSearch(unittest.TestCase):
    def test_simulation_policy(TestCase):
        search = BeamSearchTree(
            starting_state.copy(), policy, test_reward_function, 4, 2
        )
        for i in range(1):
            search.simulation_policy()

        search = BeamSearchTree(
            starting_state.copy(), policy, sequential_reward_function, 4, 2
        )

        search = BeamSearchTree(
            starting_state.copy(), policy, gpt_reward_function, 4, 2
        )
        start = time.time()
        for i in range(1):
            search.simulation_policy()
        end = time.time()

        parallel_time = end - start
        print(f"Parallel TIME: {parallel_time}")

        start = time.time()
        for i in range(1):
            search.simulation_policy()
        end = time.time()

        sequential_time = end - start
        print(f"SEQUENTIAL TIME: {sequential_time}")

        assert parallel_time > sequential_time * 2


if __name__ == "__main__":
    unittest.main()
