"""Functions and classes for querying OpenAI."""
import datetime
import logging
import os
import re

from ast import literal_eval
from copy import deepcopy
from typing import Union

import backoff

import numpy as np

import openai
from openai.embeddings_utils import get_embeddings, cosine_similarity


logging.getLogger().setLevel(logging.INFO)


class QueryState:
    """A class for the search tree state."""

    def __init__(
        self,
        template: str,
        reward_template: str,
        ads_symbols: list[str],
        ads_preferences: list[float] = None,
        catalyst_label: str = " catalysts",
        num_answers: int = 3,
        prev_candidate_list: list[str] = [],
        relation_to_candidate_list: str = None,
        include_list: list[str] = [],
        exclude_list: list[str] = [],
        answer: str = None,
        num_queries: int = 0,
        prediction_model: str = "gpt-3.5-turbo",
        reward_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        info: dict = None,
        reward: float = None,
        **kwargs,
    ):
        """Initialize the object."""
        self.template = template
        self.reward_template = reward_template
        self.ads_symbols = ads_symbols.copy()
        if ads_preferences is None:
            self.ads_preferences = [1] * len(self.ads_symbols)
        else:
            self.ads_preferences = ads_preferences.copy()
        self.catalyst_label = catalyst_label
        self.num_answers = num_answers
        self.prev_candidate_list = prev_candidate_list.copy()
        self.relation_to_candidate_list = relation_to_candidate_list
        self.include_list = include_list.copy()
        self.exclude_list = exclude_list.copy()
        self.answer = answer
        self.num_queries = num_queries
        self.prediction_model = prediction_model
        self.reward_model = reward_model
        self.embedding_model = embedding_model
        if info is not None:
            self.info = info
        else:
            self.info = {"reward": [], "generation": {}, "prior": {}}
        self.reward = reward

    def copy(self):
        """Return a copy of self."""
        return QueryState(
            template=self.template,
            reward_template=self.reward_template,
            ads_symbols=self.ads_symbols.copy(),
            ads_preferences=self.ads_preferences.copy(),
            catalyst_label=self.catalyst_label,
            prev_candidate_list=self.prev_candidate_list.copy(),
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            answer=self.answer,
            num_queries=self.num_queries,
            prediction_model=self.prediction_model,
            reward_model=self.reward_model,
            embedding_model=self.embedding_model,
            info=deepcopy(self.info),
            reward=self.reward,
        )

    def return_next(self):
        """Return a copy of self."""
        return QueryState(
            template=self.template,
            reward_template=self.reward_template,
            ads_symbols=self.ads_symbols.copy(),
            prev_candidate_list=self.candidates,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            answer=None,
            num_queries=0,
            prediction_model=self.prediction_model,
            reward_model=self.reward_model,
            embedding_model=self.embedding_model,
        )

    @property
    def prompt(self):
        """Return the prompt for this state."""
        return generate_expert_prompt(
            template=self.template,
            catalyst_label=self.catalyst_label,
            num_answers=self.num_answers,
            candidate_list=self.prev_candidate_list,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list,
            exclude_list=self.exclude_list,
        )

    @property
    def system_prompt_generation(self):
        """Return the system prompt for the generation prompt."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of catalysis. "
            "You will give recommendations for catalysts, including chemically "
            "accurate descriptions of interaction between the catalysts and adsorbates."
        )

    @property
    def system_prompt_reward(self):
        """Return the prompt for this state."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of catalysis. "
            "Particularly, given a catalyst and adsorbate, you can give a reasonable "
            "approximate of the adsorption energy, in eV."
        )

    @property
    def adsorption_energy_prompts(self):
        """Return the prompt for this state."""
        return [
            generate_adsorption_energy_list_prompt(
                ads,
                self.candidates,
            )
            for ads in self.ads_symbols
        ]

    def query(self):
        """Run a query to the LLM and change the state of self."""
        self.info["generation"] = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt_generation,
        }
        # self.answer = self.send_query(
        #     self.prompt,
        #     system_prompt=self.system_prompt_generation,
        #     model=self.prediction_model,
        # )
        self.answer = """5. Zinc oxide (ZnO):
Zinc oxide is another metal oxide catalyst that can effectively adsorb CHOHCH2. It has a high surface area and can form hydrogen bonds with CHOHCH2, facilitating its adsorption. Zinc oxide catalysts are also cost-effective and commonly used in various catalytic processes.

Finally, here is the Python list final_answer of the top-5 catalysts for the adsorption of CHOHCH2:

final_answer = ["Platinum (Pt)", "Palladium (Pd)", "Copper (Cu)", "Iron oxide (Fe2O3)", "Zinc oxide (ZnO)"]"""
        self.info["generation"] = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt_generation,
            "answer": self.answer,
        }

    @property
    def candidates(self):
        """Return the candidate list of the current answer."""

        return (
            [] if self.answer is None else parse_answer(self.answer, self.num_answers)
        )

    def query_adsorption_energy_list(self, catalyst_slice=slice(None, None)):
        """Run a query to the LLM and change the state of self."""
        r = np.random.rand(1)[0]
        self.info["reward"] = [
            {
                "prompt": self.adsorption_energy_prompts,
                "system_prompt": self.system_prompt_reward,
                "answer": ["ans"] * len(self.adsorption_energy_prompts),
                "value": r,
            }
        ]
        return r
        retries = 0
        error = None
        while retries < 3:
            retries += 1
            try:  # Try parsing out the given answer
                answers = []
                for adsorption_energy_prompt in self.adsorption_energy_prompts:
                    answer = self.send_query(
                        adsorption_energy_prompt,
                        model=self.reward_model,
                        system_prompt=self.system_prompt_reward,
                    )
                    number_answers = []
                    for line in answer.split("\n"):
                        if ":" in line:
                            _, number = line.split(":")
                            number = (
                                number.lower()
                                .replace("(ev)", "")
                                .replace("ev", "")
                                .replace(",", "")
                                .strip()
                            )
                            if (
                                re.match(r"^-?\d+(?:\.\d+)$", number) is not None
                                or number != ""
                            ):
                                number_answers.append(abs(float(number)))
                    if not len(number_answers) == len(self.candidates):
                        raise ValueError(
                            f"Found {len(number_answers)} adsorption energies. "
                            f"Expected {len(self.candidates)}."
                        )

                    answers.append(number_answers)

                output = np.mean(
                    [
                        np.mean(ans) ** (self.ads_preferences[i])
                        for i, ans in enumerate(answers)
                    ]
                )
                return output
            except Exception as err:
                error = err
                logging.warning(
                    f"Failed to parse answer with error: {err}. Generating new answer."
                )
                self.query()
        raise error

    def send_query(self, prompt, model=None, system_prompt=None):
        """Send the query to OpenAI and increment."""
        if model is None:
            model = self.prediction_model
        answer = run_query(prompt, model=model, system_prompt=system_prompt)
        self.num_queries += 1
        return answer

    def similarity(self, states: "list[QueryState]") -> float:
        """Calculate a similarity score of this state with a list of trial states."""
        relevant_strings = [self.prompt, self.answer]
        if any([s is None for s in relevant_strings]):
            return np.ones(len(states), dtype=float)

        for state in states:
            relevant_strings.append(state.prompt)
        # embeddings = run_get_embeddings(relevant_strings, model=self.embedding_model)
        embeddings = [np.random.rand(356) for _ in range(len(relevant_strings))]
        self.info["priors"] = {"embeddings": embeddings}
        p = embeddings.pop(0)
        y = embeddings.pop(0)
        p_y = np.array(p) + np.array(y)
        similarities = []
        while len(embeddings) > 0:
            similarities.append(cosine_similarity(embeddings.pop(0), p_y))

        similarities = np.array(similarities)
        self.info["priors"].update({"similarities": similarities})
        return similarities + (1 - similarities)

    def set_reward(self, r: float, metadata: dict = None):
        """Set the reward for this state."""
        self.reward = r
        if metadata is not None:
            self.info["reward"] = metadata


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
            "tot ehir adsorption energies."
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
        candidate_list_statement = f"{relation_to_candidate_list} "
        candidate_list_statement += ", ".join(candidate_list).strip() + " "
    elif len(candidate_list) != 0 and relation_to_candidate_list is None:
        raise ValueError(
            f"Non-empty candidate list {candidate_list} given with "
            "relation_to_candidate_list == None"
        )
    else:
        candidate_list_statement = ""
    if len(include_list) != 0:
        include_statement = (
            f"Include candidate{catalyst_label} with the following properties: "
        )
        include_statement += ", ".join(include_list)
        include_statement += ". "
    else:
        include_statement = ""
    if len(exclude_list) != 0:
        exclude_statement = (
            f"Exclude candidate{catalyst_label} with the following properties: "
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
    list_start = answer.find("[", final_answer_location)
    list_end = answer.find("]", list_start)
    try:
        answer_list = literal_eval(answer[list_start : list_end + 1])  # noqa:E203
    except Exception:
        answer_list = answer[list_start + 1 : answer.find("]", list_start)]  # noqa:E203
        answer_list = [ans.replace("'", "") for ans in answer_list.split(",")]
    return [ans.replace('"', "").replace("'", "").strip() for ans in answer_list]


def init_openai():
    """Initialize connection to OpenAI."""
    openai.api_key = os.getenv("OPENAI_API_KEY_DEV")
    return


query_counter = 0
tok_sent = 0
tok_recieved = 0


@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=60)
def run_get_embeddings(strings, model="text-embedding-ada-002"):
    """Query language model for a list of k candidates."""
    return get_embeddings(strings, engine=model)


@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_time=60)
def run_query(query, model="gpt-3.5-turbo", system_prompt=None, **gpt_kwargs):
    """Query language model for a list of k candidates."""
    gpt_kwargs["temperature"] = gpt_kwargs.get("temperature", 0.6)
    gpt_kwargs["top_p"] = gpt_kwargs.get("top_p", 1.0)
    gpt_kwargs["max_tokens"] = gpt_kwargs.get("max_tokens", 1300)
    now = datetime.datetime.now()
    logging.info(f"New query at time: {now}")

    # output = openai.Completion.create(
    #     model="text-davinci-003", max_tokens=1300, temperature=1, prompt=query
    # )

    if model == "text-davinci-003":
        output = openai.Completion.create(model=model, prompt=query, **gpt_kwargs)
        answer = output["choices"][0]["text"]
    elif "gpt-3.5" in model or "gpt-4" in model:
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": query})
        output = openai.ChatCompletion.create(
            model=model, messages=messages, **gpt_kwargs
        )
        answer = output["choices"][0]["message"]["content"]

    logging.info(f"--------------------\nQ: {query}\n--------------------")

    global query_counter
    query_counter += 1
    logging.info(f"Num queries run: {query_counter}")

    global tok_sent
    tok_sent += output["usage"]["prompt_tokens"]
    logging.info(f"Total num tok sent: {tok_sent}")

    now = datetime.datetime.now()
    logging.info(f"Answer recieved at time: {now}")

    logging.info(f"--------------------\nA: {answer}\n--------------------")

    global tok_recieved
    tok_recieved += output["usage"]["completion_tokens"]
    logging.info(f"Total num tok recieved: {tok_recieved}\n\n")

    return answer


init_openai()
