"""Functions and classes for querying OpenAI."""
import asyncio
import datetime
import logging
import os
import re
import sys
import time

from ast import literal_eval
from copy import deepcopy

import backoff

import numpy as np

import openai

sys.path.append("src")


logging.getLogger().setLevel(logging.INFO)


# _reward_system_prompt = "You are a helpful catalysis expert with extensive knowledge "
#     "of the adsorption of atoms and molecules. You can offer an approximate value of "
#     "adsorption energies of various adsorbates to various catalysts."


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f'f"{fstring_text}"', vals)
    return ret_val


openai_client = None


def init_openai():
    """Initialize connection to OpenAI."""
    global openai_client
    if openai_client is None:
        openai_client = openai.AsyncOpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY_DEV")


llama_model = None


def init_llama(
    model_dir="meta-llama/Llama-2-13b-chat-hf", num_gpus=1, **kwargs_sampling_params
):
    """Initialize the llama model and load in on the gpu."""
    global llama_model
    if llama_model is None:
        from src.llm.llama2_vllm_chemreasoner import LlamaLLM

        llama_model = LlamaLLM(model_dir=model_dir, num_gpus=num_gpus)


query_counter = 0
tok_sent = 0
tok_recieved = 0


async def parallel_openai_chat_completion(
    prompt, system_prompt=None, model="gpt-3.5-turbo", **kwargs
):
    """Run chat completion calls on openai, in parallel."""
    global openai_client
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return await openai_client.chat.completions.create(
        messages=messages, model=model, **kwargs
    )


async def openai_chat_async_evaluation(prompts, system_prompts, model, **kwargs):
    completions = [
        parallel_openai_chat_completion(p, system_prompts[i])
        for i, p in enumerate(prompts)
    ]

    answers = await asyncio.gather(*completions)
    return answers


# @backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=120)
def run_prompts(
    prompts,
    system_prompts=None,
    model="gpt-3.5-turbo",
    max_pause=0,
    **kwargs,
):
    """Query language model for a list of k candidates."""
    kwargs["temperature"] = kwargs.get("temperature", 0.6)
    kwargs["top_p"] = kwargs.get("top_p", 0.3)
    kwargs["max_tokens"] = kwargs.get("max_tokens", 1300)
    now = datetime.datetime.now()
    logging.info(f"New query at time: {now}")

    if system_prompts is None:
        system_prompts = [None] * len(prompts)

    # output = openai.Completion.create(
    #     model="text-davinci-003", max_tokens=1300, temperature=1, prompt=query
    # )

    if "gpt-3.5" in model or "gpt-4" in model:
        init_openai()
        answer_objects = asyncio.run(
            openai_chat_async_evaluation(
                prompts,
                system_prompts=system_prompts,
                model="gpt-3.5-turbo",
                **kwargs,
            )
        )
        answer_strings = [a.choices[0].message.content for a in answer_objects]
        return answer_strings

    elif "llama" in model:
        init_llama()
        global llama_model

        answer_strings = llama_model(prompts, system_prompts, **kwargs)
        return answer_strings

    else:
        raise ValueError(f"Unkown model: {model}")


llama_generator = None


if __name__ == "__main__":
    # llama_key = os.getenv("LLAMA_KEY")
    # login(llama_key)
    # llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    # llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

    # logging.info(
    #     llama_get_embeddings(
    #         ["This is the query, is it?", "Making a western movie is a cowboy thing."],
    #     )
    # )

    messages = [{"role": "user", "content": "Say this is a test"} for _ in range(3)]

    from openai import AsyncOpenAI

    init_openai()

    client = AsyncOpenAI()

    # help(client.chat.completions.create)

    # async def test():
    #     for message in messages:
    #         stream = await client.chat.completions.create(
    #             prompt="Say this is a test",
    #             messages=[{"role": "user", "content": "Say this is a test"}],
    #             stream=True,
    #         )

    #     async for part in stream:
    #         print(part.choices[0].delta.content or "")

    # import asyncio

    # async def async_myfunc(dictionary):
    #     # Your async function logic here
    #     # This is just a placeholder, replace it with your actual async function
    #     await asyncio.sleep(1)  # Simulating an asynchronous operation
    #     return dictionary["value"] * 2  # Replace with your actual logic

    # async def async_process_input_list(input_list):
    #     tasks = [async_myfunc(dictionary) for dictionary in input_list]
    #     results = await asyncio.gather(*tasks)
    #     return results

    # # Example usage
    # input_list = [
    #     {"role": "user", "content": "Say this is a test"},
    #     {"role": "user", "content": "This is another test"},
    #     {"role": "user", "content": "Final test"},
    # ]

    # # Run the event loop
    # loop = asyncio.get_event_loop()
    # result = loop.run_until_complete(async_process_input_list(input_list))

    print("testing run prompts")

    print(run_prompts(["test1", "test2", "test3"], model="gpt-3.5-turbo"))
