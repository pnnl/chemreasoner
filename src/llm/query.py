"""Functions and classes for querying OpenAI."""
import asyncio
import datetime
import logging
import os
import re
import time

from ast import literal_eval
from copy import deepcopy

import backoff
import torch

from huggingface_hub import login
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline

import openai

logging.getLogger().setLevel(logging.INFO)


# _reward_system_prompt = "You are a helpful catalysis expert with extensive knowledge "
#     "of the adsorption of atoms and molecules. You can offer an approximate value of "
#     "adsorption energies of various adsorbates to various catalysts."


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f'f"{fstring_text}"', vals)
    return ret_val


global openai_client
openai_client = None


def init_openai():
    """Initialize connection to OpenAI."""
    global openai_client
    if openai_client is None:
        openai_client = openai.AsyncOpenAI()
        openai.api_key = os.getenv("OPENAI_API_KEY_DEV")


query_counter = 0
tok_sent = 0
tok_recieved = 0


@backoff.on_exception(backoff.expo, Exception, max_time=120)
def run_get_embeddings(strings, model="text-embedding-ada-002"):
    """Query language model for a list of k candidates."""
    if model == "text-embedding-ada-002":
        return get_embeddings(strings, engine=model)
    elif "llama" in model:
        return llama_get_embeddings(strings)


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

    if model == "text-davinci-003":
        init_openai()
        random_wait = np.random.randint(low=0, high=max_pause + 1)
        time.sleep(random_wait)
        output = openai.Completion.create(model=model, prompt=prompts, **kwargs)
        answer = output["choices"][0]["text"]
    elif "gpt-3.5" in model or "gpt-4" in model:
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
        global llama_generator
        sys_prompt = "" if system_prompt is None else system_prompt
        answer = generate_cand(llama_generator, sys_prompt, query)
    logging.info(f"--------------------\nQ: {query}\n--------------------")

    global query_counter
    query_counter += 1
    logging.info(f"Num queries run: {query_counter}")

    if "llama" not in model:
        global tok_sent
        tok_sent += output["usage"]["prompt_tokens"]
        logging.info(f"Total num tok sent: {tok_sent}")

    now = datetime.datetime.now()
    logging.info(f"Answer recieved at time: {now}")

    logging.info(f"--------------------\nA: {answer}\n--------------------")

    if "llama" not in model:
        global tok_recieved
        tok_recieved += output["usage"]["completion_tokens"]
        logging.info(f"Total num tok recieved: {tok_recieved}\n\n")

    return answer


llama_generator = None


def get_device():
    """Get proper device for llama."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    else:
        return "cpu"


def init_llama(llama_weights="meta-llama/Llama-2-13b-chat-hf"):
    """Initialize the llama model and load in on the gpu."""
    device = get_device()
    global llama_generator  # , llama_model, llama_tokenizer
    # if llama_model is None:

    #     llama_model = LlamaForCausalLM.from_pretrained(llama_weights)
    #     llama_model.to(device)
    if llama_generator is None:
        llama_key = os.getenv("LLAMA_KEY")
        login(llama_key)
        llama_generator = pipeline(model=llama_weights, device=device)

    # if llama_tokenizer is None:
    #     llama_tokenizer = LlamaTokenizer(model=llama_weights)


def generate_cand(generator, sys_prompt, user_prompt):
    # sys_prompt =  prompt['generation_prompt']['system']
    # user_prompt = prompt['generation_prompt']['user']
    gen_prompt = (
        "<s>[INST] <<SYS>>\n"
        + sys_prompt
        + "\n<</SYS>>"
        + "\n\n"
        + user_prompt
        + " [/INST]"
    )
    # print(gen_prompt)
    answer = generator(gen_prompt)
    return answer[0]["generated_text"].split("[/INST]")[-1]


def llama_get_embeddings(strings):
    """Get the embeddings with the given llama model."""

    global llama_model, llama_tokenizer
    input_ids = torch.tensor(llama_tokenizer.encode(strings)).unsqueeze(0)
    logging.info(f"Input_ids:\n{input_ids}")
    outputs = llama_model(input_ids)
    logging.info(f"model_outputs:\n{outputs}")
    last_hidden_states = outputs[0]
    logging.info(f"last_hidden_states:\n{last_hidden_states}")
    return last_hidden_states


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
