from unittest import TestCase


class TestBeamSearch(TestCase):
    def test_simulation_policy(TestCase):
        if len(prompts) > 1:
            print("\n*" * 5)
        elif len(prompts) == 1:
            print("\n------" * 5)
        else:
            print("\n~~~~~~~" * 5)
        return [p + s for p, s in zip(prompts, system_prompts)]

        reward_function = LLMRewardFunction(
            llm_function=llm_function,
        )

        search = BeamSearchTree(starting_state, policy, reward_function, 4, 3)

        for i in range(3):
            search.simulation_policy()

        n_batch = 3
        states = [starting_state.copy() for _ in range(n_batch)]
        # print(reward_function(states))
        assert len(states) == 3
