"""Create a class for a reasoner state."""
import logging
import re
import time

from ast import literal_eval
from copy import deepcopy
from typing import Union

import numpy as np

logging.getLogger().setLevel(logging.INFO)


_example_generation_answer = """5. Zinc oxide (ZnO):
Zinc oxide is another metal oxide catalyst that can effectively adsorb CHOHCH2. It has a high surface area and can form hydrogen bonds with CHOHCH2, facilitating its adsorption. Zinc oxide catalysts are also cost-effective and commonly used in various catalytic processes.

Finally, here is the Python list final_answer of the top-5 catalysts for the adsorption of CHOHCH2:

final_answer = ["Platinum (Pt)", "Palladium (Pd)", "Copper (Cu)", "Iron oxide (Fe2O3)", "Zinc oxide (ZnO)"]"""


class ReasonerState:
    """A class for the search tree state."""

    def __init__(
        self,
        template: str,
        reward_template: str,
        ads_symbols: list[str],
        ads_preferences: list[float] = None,
        catalyst_label: str = "catalysts",
        num_answers: int = 3,
        prev_candidate_list: list[str] = [],
        relation_to_candidate_list: str = None,
        include_list: list[str] = [],
        exclude_list: list[str] = [],
        answer: str = None,
        embeddings: dict = {},
        num_queries: int = 0,
        query_time: float = 0.0,
        info: dict = None,
        reward: float = None,
        debug=False,
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
        self.embeddings = embeddings
        self.num_queries = num_queries
        self.query_time = query_time
        if info is not None:
            self.info = info
        else:
            self.info = {"generation": {}, "priors": {}}
        self.reward = reward
        self.debug = debug

    @classmethod
    @staticmethod
    def from_dict(data: dict):  # TODO: Add defaults
        """Create a query state from dictionary."""
        return ReasonerState(
            template=data.get("template"),
            reward_template=data.get("reward_template"),
            ads_symbols=data.get("ads_symbols").copy(),
            ads_preferences=data.get("ads_preferences", None),
            catalyst_label=data.get("catalyst_label"),
            prev_candidate_list=data.get("prev_candidate_list", []).copy(),
            relation_to_candidate_list=data.get("relation_to_candidate_list", None),
            include_list=data.get("include_list", []).copy(),
            exclude_list=data.get("exclude_list", []).copy(),
            answer=data.get("answer", None),
            embeddings=data.get("embeddings", {}).copy(),
            num_queries=data.get("num_queries", 0),
            query_time=data.get("query_time", 0.0),
            info=deepcopy(data.get("self.info", {})),
            reward=data.get("reward", None),
            debug=data.get("debug", False),
        )

    def copy(self):
        """Return a copy of self."""
        return ReasonerState(
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
            embeddings=self.embeddings.copy(),
            num_queries=self.num_queries,
            query_time=self.query_time,
            info=deepcopy(self.info),
            reward=self.reward,
            debug=self.debug,
        )

    def return_next(self) -> "ReasonerState":
        """Return the successor state of self."""
        return ReasonerState(
            template=self.template,
            reward_template=self.reward_template,
            ads_symbols=self.ads_symbols.copy(),
            prev_candidate_list=self.candidates,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            answer=None,
            embeddings={},
            num_queries=0,
            query_time=0.0,
            debug=self.debug,
        )

    @property
    def generation_prompt(self):
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
    def generation_system_prompt(self):
        """Return the system prompt for the generation prompt."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of "
            "catalysis. You will give recommendations for catalysts, including "
            "chemically accurate descriptions of the interaction between the catalysts "
            "and adsorbate(s). Make specific recommendations for catalysts, including "
            "their chemical composition. Make sure to follow the formatting "
            "instructions. Do not provide disclaimers or notes about your knowledge of "
            "catalysis."
        )

    @property
    def reward_system_prompt(self):
        """Return the prompt for this state."""
        return (
            "You are a helpful chemistry expert with extensive knowledge of catalysis. "
            "Particularly, given a catalyst and adsorbate, you can give a reasonable "
            "approximation of the adsorption energy, in eV."
        )

    @property
    def candidates(self):
        """Return the candidate list of the current answer."""
        return (
            [] if self.answer is None else parse_answer(self.answer, self.num_answers)
        )

    def process_generation(
        self, results  # ={"answer": _example_generation_answer, "usage": 0}
    ):
        """process generation answer and store."""
        self.answer = results["answer"]
        self.info["generation"] = {
            "prompt": self.generation_prompt,
            "system_prompt": self.generation_system_prompt,
            "answer": self.answer,
            "candidates_list": self.candidates,
            "usage": results["usage"],
        }
        print(self.candidates)
        return True

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

    def process_adsorption_energy(
        self, results  # =None[{"answer": _example_generation_answer, "usage": 0}]*len()
    ):
        """Process the return adsorption energy answers into values and store."""
        if "llm_reward" not in self.info:
            self.info["llm-reward"] = {"attempted_prompts": []}

        self.info["llm-reward"]["attempted_prompts"].append(
            {
                "prompt": self.adsorption_energy_prompts,
                "system_prompt": self.reward_system_prompt,
                "answer": [],
                "key_answers": [],
                "number_answers": [],
                "successful": [],
                "usage": [],
            }
        )
        return_values = []
        for i, adsorption_energy_prompt in enumerate(self.adsorption_energy_prompts):
            ans = results[i]["answer"]
            # store the answer
            self.info["llm-reward"]["attempted_prompts"][-1]["answer"].append(ans)
            self.info["llm-reward"]["attempted_prompts"][-1]["usage"].append(
                results[i]["usage"]
            )

            key_answers = []
            number_answers = []
            try:
                for line in ans.split("\n"):
                    if ":" in line:
                        k, number = line.split(":")
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
                        key_answers.append(k)
                        if not len(number_answers) == len(self.candidates):
                            raise ValueError(
                                f"Found {len(number_answers)} adsorption energies. "
                                f"Expected {len(self.candidates)}."
                            )

                        # store key_answers and number_answers
                        self.info["llm-reward"]["attempted_prompts"][-1][
                            "key_answers"
                        ].append(key_answers)
                        self.info["llm-reward"]["attempted_prompts"][-1][
                            "number_answers"
                        ].append(number_answers)
                        # Save the return values
                        return_values.append(number_answers)
            except Exception as err:
                # Save and rerase error
                self.info["llm-reward"]["attempted_prompts"][-1]["key_answers"].append(
                    key_answers
                )
                self.info["llm-reward"]["attempted_prompts"][-1][
                    "number_answers"
                ].append(number_answers)
                raise err

        return return_values

    @property
    def catalyst_symbols_prompt(self):
        """Turn this state's answer into a prompt for symbols_parsing."""
        example_format = ""
        for i, ans in enumerate(self.candidates):
            example_format += f"{ans}: [list_{i}]\n"

        answer_string = ", ".join(self.candidates)
        prompt = (
            f"Consider the following list of catalysts:\n{answer_string}.\n\n"
            "For each catalyst, return the list of chemical symbols that make up the "
            "catalyst. If a catalyst does not have a chemical symbol, return None. "
            "If a catalyst is already a chemical formula, repeat the elements in the "
            "chemical formula.\n\n"
            "Format your list as:\n"
            f"{example_format}"
        )
        return prompt

    def process_catalyst_symbols(self, answer):
        """Turn parse out the symbols from the llm answer."""
        answer_list_parsed = [None] * len(answer)
        for line in answer.split("\n"):
            if ":" in line:
                cat, syms = line.split(":")
                idx = self.candidates.index(cat)  # ensure ording is preserved
                syms_list = list(
                    {s.strip() for s in syms.strip().strip("[").strip("]").split(",")}
                )  # Use a set for unique elements only
                if syms_list == ["None"]:
                    syms_list = None
                answer_list_parsed[idx] = syms_list

        return answer_list_parsed

    def query_adsorption_energy_list(
        self,
        allow_requery=True,
    ):
        """Run a query with the LLM and change the state of self."""
        self.info["llm-reward"] = {"attempted_prompts": []}

        retries = 0
        error = None
        while retries < 3:
            retries += 1
            try:  # Try parsing out the given answer
                if allow_requery:
                    self.query()
                self.info["llm-reward"]["attempted_prompts"].append(
                    {
                        "prompt": self.adsorption_energy_prompts,
                        "system_prompt": self.reward_system_prompt,
                        "answer": [],
                        "key_answers": [],
                        "number_answers": [],
                        "successful": [],
                    }
                )
                answers = []
                for adsorption_energy_prompt in self.adsorption_energy_prompts:
                    if self.debug:
                        self.info["llm-reward"]["attempted_prompts"][retries - 1][
                            "answer"
                        ].append("ans")
                        answers.append(list(np.random.rand(3)))
                    else:
                        answer = self.send_query(
                            adsorption_energy_prompt,
                            system_prompt=self.reward_system_prompt,
                        )
                        self.info["llm-reward"]["attempted_prompts"][retries - 1][
                            "answer"
                        ].append(answer)
                        key_answers = []
                        number_answers = []
                        for line in answer.split("\n"):
                            if ":" in line:
                                k, number = line.split(":")
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
                                key_answers.append(k)
                        self.info["llm-reward"]["attempted_prompts"][retries - 1][
                            "key_answers"
                        ].append(key_answers)
                        self.info["llm-reward"]["attempted_prompts"][retries - 1][
                            "number_answers"
                        ].append(number_answers)
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
                    f"Failed to parse answer with error: {str(err)}. Generating new answer."
                )
        logging.warning(
            f"Failed to parse answer with error: {str(error)}. Returning a penalty value."
        )
        return -10

    # def send_query(self, prompt, model=None, system_prompt=None):
    #     """Send the query to OpenAI and increment."""
    #     if model is None:
    #         model = self.prediction_model
    #     start = time.time()
    #     answer = run_query(prompt, model=model, system_prompt=system_prompt)
    #     end = time.time()
    #     self.query_time += end - start
    #     self.num_queries += 1
    #     return answer

    def similarity(self, states: "list[ReasonerState]") -> float:
        """Calculate a similarity score of this state with a list of trial states."""
        if (
            "prompt" not in self.embeddings.keys()
            or "answer" not in self.embeddings.keys()
        ):
            return np.ones(len(states), dtype=float)
        else:
            embeddings = [self.embeddings["prompt"], self.embeddings["answer"]]

            relevant_strings = []
        for state in states:
            relevant_strings.append(state.generation_prompt)
        if not self.debug:
            embeddings += run_get_embeddings(
                relevant_strings, model=self.embedding_model
            )
        else:
            embeddings += [np.random.rand(356) for _ in range(len(relevant_strings))]

        p = embeddings.pop(0)
        y = embeddings.pop(0)
        p_y = np.array(p) + np.array(y)

        self.info["priors"].update({"embeddings": embeddings.copy()})
        similarities = []
        while len(embeddings) > 0:
            similarities.append(cosine_similarity(embeddings.pop(0), p_y))

        similarities = np.array(similarities)
        self.info["priors"].update({"similarities": similarities})
        return similarities + (1 - similarities)

    def set_reward(self, r: float, primary_reward: bool = True, info_field: str = None):
        """Set the reward for this state."""
        if primary_reward is None:
            self.reward = r
        if info_field is not None:
            if info_field in self.info.keys():
                self.info[info_field]["value"] = r
            else:
                self.info[info_field] = {"value": r}


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


def fstr(fstring_text, vals):
    """Evaluate the provided fstring_text."""
    ret_val = eval(f'f"{fstring_text}"', vals)
    return ret_val


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
