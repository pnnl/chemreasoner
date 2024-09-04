"""General utilities for LLM functions."""

from typing import Optional


def process_prompt(
    llm_function: callable,
    prompt_info: list[str],
    prompt_creation_function: callable,
    prompt_parsing_function: callable,
    system_prompt_function: callable,
    retries: Optional[int] = 0,
    **llm_function_kwargs,
):
    """A script to handle the creation, running, parsing of a prompt."""
    return_value = False
    if not isinstance(prompt_info, list):
        return_value = True
        prompt_info = [prompt_info]

    attempts = 0
    answers = [None] * len(prompt_info)
    while attempts <= retries and any([ans is None for ans in answers]):
        # Get the prompts
        prompt_idx = []
        prompts = []
        system_prompts = []
        for i, p_info in enumerate(prompt_info):
            if answers[i] is None:  # If answer hasn't been found, yet
                proposed_prompt = prompt_creation_function(p_info)
                if proposed_prompt is not None:  # If prompt creation is successful
                    prompt_idx.append(i)
                    prompts.append(prompt_creation_function(p_info))
                    system_prompts.append(system_prompt_function(p_info))
        print(prompts)
        # Run non-None prompts through the LLM to get raw answers
        raw_answers = llm_function(prompts, system_prompts, **llm_function_kwargs)
        print(raw_answers)
        for p_idx, raw_answer in zip(prompt_idx, raw_answers):
            # Attempt to parse out the answer
            try:
                answers[p_idx] = prompt_parsing_function(raw_answer, prompt_info[p_idx])
            except Exception:
                answers[p_idx] = None
        attempts += 1
    print(answers)
    if return_value:
        return answers[0]
    else:
        return answers
