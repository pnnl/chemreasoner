"""Functions and classes for querying OpenAI."""
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
from openai.embeddings_utils import get_embeddings, cosine_similarity


logging.getLogger().setLevel(logging.INFO)


# _reward_system_prompt = "You are a helpful catalysis expert with extensive knowledge "
#     "of the adsorption of atoms and molecules. You can offer an approximate value of "
#     "adsorption energies of various adsorbates to various catalysts."


def generate_adsorption_energy_list_prompt(
    adsorbate: str, candidate_list: list[str], reward_template: str = None
):
    """Make a query to get a list of adsorption energies."""
    if reward_template is None:
        prompt = (
            "Generate a list of adsorption energies, in eV, "
            f"for the adsorbate {adsorbate} to the surface of "
            f"each of the following catalysts: {', '.join(candidate_list)}. "
            f"Return your answer as a python dictionary mapping catalysts "
            "to their adsorption energies."
        )
    else:
        vals = {"adsorbate": adsorbate, "candidate_list": candidate_list}
        prompt = fstr(reward_template, vals)
    return prompt


def generate_expert_prompt(
    template: str,
    catalyst_label: str,
    num_answers: int,
    candidate_list: list = [],
    relation_to_candidate_list: str = None,
    include_list: list = [],
    exclude_list: list = [],
):
    """Generate prompt based on catalysis experts."""
    if len(candidate_list) != 0 and relation_to_candidate_list is not None:
        candidate_list_statement = "\n\nYou should start with the following list: "
        candidate_list_statement += (
            "["
            + ", ".join(
                [
                    "'" + cand.replace("'", "").replace('"', "").strip() + "'"
                    for cand in candidate_list
                ]
            )
            + "]. "
        )
        candidate_list_statement += "The list that you return should probably not have the same catalysts as this list! "
        candidate_list_statement += f"Your list of {catalyst_label} may {relation_to_candidate_list} those in the list. "
        candidate_list_statement += (
            "Please compare your list to some of the candidates in this list."
        )
    elif len(candidate_list) != 0 and relation_to_candidate_list is None:
        raise ValueError(
            f"Non-empty candidate list {candidate_list} given with "
            "relation_to_candidate_list == None"
        )
    else:
        candidate_list_statement = ""
    if len(include_list) != 0:
        include_statement = (
            f"You should include candidate {catalyst_label} "
            "with the following properties: "
        )
        include_statement += ", ".join(include_list)
        include_statement += ". "
    else:
        include_statement = ""
    if len(exclude_list) != 0:
        exclude_statement = (
            f"You should exclude candidate {catalyst_label} "
            "with the following properties: "
        )

        exclude_statement += ", ".join(exclude_list)
        exclude_statement += ". "
    else:
        exclude_statement = ""
    vals = {
        "catalyst_label": catalyst_label,
        "candidate_list_statement": candidate_list_statement,
        "include_statement": include_statement,
        "exclude_statement": exclude_statement,
    }
    return fstr(template, vals)


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f'f"{fstring_text}"', vals)
    return ret_val


def parse_answer(answer: str, num_expected=None):
    """Parse an answer into a list."""
    final_answer_location = answer.lower().find("final_answer")
    if final_answer_location == -1:
        final_answer_location = answer.lower().find("final answer")
    if final_answer_location == -1:
        final_answer_location = answer.lower().find("final")  # last ditch effort
    if final_answer_location == -1:
        final_answer_location = 0
    list_start = answer.find("[", final_answer_location)
    list_end = answer.find("]", list_start)
    try:
        answer_list = literal_eval(answer[list_start : list_end + 1])  # noqa:E203
    except Exception:
        answer_list = answer[list_start + 1 : list_end]  # noqa:E203
        answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
    return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]


def init_openai():
    """Initialize connection to OpenAI."""
    openai.api_key = os.getenv("OPENAI_API_KEY_DEV")
    return


query_counter = 0
tok_sent = 0
tok_recieved = 0


@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=120)
def run_get_embeddings(strings, model="text-embedding-ada-002"):
    """Query language model for a list of k candidates."""
    if model == "text-embedding-ada-002":
        return get_embeddings(strings, engine=model)
    elif "llama" in model:
        return llama_get_embeddings(strings)


# @backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=120)
def run_query(
    query, model="gpt-3.5-turbo", system_prompt=None, max_pause=0, **gpt_kwargs
):
    """Query language model for a list of k candidates."""
    gpt_kwargs["temperature"] = gpt_kwargs.get("temperature", 0.6)
    gpt_kwargs["top_p"] = gpt_kwargs.get("top_p", 0.3)
    gpt_kwargs["max_tokens"] = gpt_kwargs.get("max_tokens", 1300)
    now = datetime.datetime.now()
    logging.info(f"New query at time: {now}")

    # output = openai.Completion.create(
    #     model="text-davinci-003", max_tokens=1300, temperature=1, prompt=query
    # )

    if model == "text-davinci-003":
        random_wait = np.random.randint(low=0, high=max_pause + 1)
        time.sleep(random_wait)
        output = openai.Completion.create(model=model, prompt=query, **gpt_kwargs)
        answer = output["choices"][0]["text"]
    elif "gpt-3.5" in model or "gpt-4" in model:
        random_wait = np.random.randint(low=0, high=max_pause + 1)
        time.sleep(random_wait)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": query})
        output = openai.ChatCompletion.create(
            model=model, messages=messages, **gpt_kwargs
        )
        answer = output["choices"][0]["message"]["content"]
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


init_openai()

if __name__ == "__main__":
    llama_key = os.getenv("LLAMA_KEY")
    login(llama_key)
    llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

    logging.info(
        llama_get_embeddings(
            ["This is the query, is it?", "Making a western movie is a cowboy thing."],
        )
    )
