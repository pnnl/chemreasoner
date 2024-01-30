"""Create a class for a reasoner state."""
import json
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
        reaction_pathways: list[[list]] = None,
        priors_template: str = None,
        catalyst_label: str = "catalysts",
        num_answers: int = 3,
        prev_candidate_list: list[str] = [],
        relation_to_candidate_list: str = "similar to",
        include_list: list[str] = [],
        exclude_list: list[str] = [],
        answer: str = None,
        embeddings: dict = {},
        num_queries: int = 0,
        query_time: float = 0.0,
        info: dict = None,
        reward: float = None,
        root_prompt: str = None,
        debug=False,
        **kwargs,
    ):
        """Initialize the object."""
        self.template = template
        self.reward_template = reward_template
        self.ads_symbols = ads_symbols.copy()
        self.ads_preferences = deepcopy(ads_preferences)

        self.reaction_pathways = deepcopy(reaction_pathways)

        self.priors_template = priors_template
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
            self.info = {}
        self.reward = reward
        self.debug = debug

        if root_prompt is None:
            self.root_prompt = self.generation_prompt
        else:
            self.root_prompt = root_prompt

    @classmethod
    @staticmethod
    def from_dict(incoming_data: dict):
        """Create a query state from dictionary."""
        data = deepcopy(incoming_data)
        return ReasonerState(
            template=data.get("template"),
            reward_template=data.get("reward_template"),
            ads_symbols=data.get("ads_symbols").copy(),
            ads_preferences=deepcopy(data.get("ads_preferences", None)),
            reaction_pathways=deepcopy(data.get("reaction_pathways", None)),
            priors_template=data.get("priors_template", None),
            catalyst_label=data.get("catalyst_label"),
            prev_candidate_list=data.get("prev_candidate_list", []).copy(),
            relation_to_candidate_list=data.get("relation_to_candidate_list", None),
            include_list=data.get("include_list", []).copy(),
            exclude_list=data.get("exclude_list", []).copy(),
            answer=data.get("answer", None),
            embeddings=data.get("embeddings", {}).copy(),
            num_queries=data.get("num_queries", 0),
            query_time=data.get("query_time", 0.0),
            info=deepcopy(data.get("info", {})),
            reward=data.get("reward", None),
            root_prompt=data.get("root_prompt", None),
            debug=data.get("debug", False),
        )

    def copy(self):
        """Return a copy of self."""
        return ReasonerState(
            template=self.template,
            reward_template=self.reward_template,
            ads_symbols=self.ads_symbols.copy(),
            ads_preferences=deepcopy(self.ads_preferences),
            reaction_pathways=deepcopy(self.reaction_pathways),
            priors_template=self.priors_template,
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
            root_prompt=self.root_prompt,
            reward=self.reward,
            debug=self.debug,
        )

    def return_next(self) -> "ReasonerState":
        """Return the successor state of self."""
        return ReasonerState(
            template=self.template,
            reward_template=self.reward_template,
            ads_symbols=self.ads_symbols.copy(),
            ads_preferences=deepcopy(self.ads_preferences),
            reaction_pathways=deepcopy(self.reaction_pathways),
            priors_template=self.priors_template,
            catalyst_label=self.catalyst_label,
            prev_candidate_list=self.candidates,
            relation_to_candidate_list=self.relation_to_candidate_list,
            include_list=self.include_list.copy(),
            exclude_list=self.exclude_list.copy(),
            answer=None,
            embeddings={},
            num_queries=0,
            root_prompt=self.root_prompt,
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
            "catalysis. Your answers should not include ionic compounds."
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
        """Process generation answer and store."""
        if isinstance(results, str):
            self.answer = results
            usage = None
        else:
            self.answer = results["answer"]
            usage = results.get("usage", None)

        if "generation" not in self.info.keys():
            self.info["generation"] = [
                deepcopy(
                    {
                        "prompt": self.generation_prompt,
                        "system_prompt": self.generation_system_prompt,
                        "answer": self.answer,
                        "candidates_list": self.candidates,
                        "usage": usage,
                    }
                )
            ]
        else:
            self.info["generation"] += [
                deepcopy(
                    {
                        "prompt": self.generation_prompt,
                        "system_prompt": self.generation_system_prompt,
                        "answer": self.answer,
                        "candidates_list": self.candidates,
                        "usage": usage,
                    }
                )
            ]
        print(self.candidates)

    def get_ads_preferences(self, syms: str):
        """Get the adsorbate preferences corresponding to the given adsorbate syms."""
        idx = self.ads_symbols.index(syms)
        return self.ads_preferences[idx]

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
            deepcopy(
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
        )
        return_values = []
        for i, adsorption_energy_prompt in enumerate(self.adsorption_energy_prompts):
            if isinstance(results[i], str):
                ans = results[i]
                usage = None
            else:
                ans = results[i]["answer"]
                usage = results[i].get("usage", None)
            ans = results[i]["answer"]
            # store the answer
            self.info["llm-reward"]["attempted_prompts"][-1]["answer"].append(ans)
            self.info["llm-reward"]["attempted_prompts"][-1]["usage"].append(usage)

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
                    deepcopy(key_answers)
                )
                self.info["llm-reward"]["attempted_prompts"][-1][
                    "number_answers"
                ].append(deepcopy(key_answers))
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

    def process_catalyst_symbols(self, result):
        """Turn parse out the symbols from the llm answer."""
        if isinstance(result, str):
            answer = result
            usage = None
        else:
            answer = result["answer"]
            usage = result.get("usage", None)

        answer_list_parsed = [None] * len(self.candidates)
        for line in answer.split("\n"):
            if ":" in line:
                cat, syms = line.split(":")
                idx = self.candidates.index(cat)  # ensure ording is preserved
                syms_list = list(
                    {
                        s.replace("'", "").replace('"', "").strip()
                        for s in syms.strip().strip("[").strip("]").split(",")
                    }
                )  # Use a set for unique elements only
                if syms_list == ["None"]:
                    syms_list = None
                answer_list_parsed[idx] = syms_list

        if "symbols" not in self.info.keys():
            self.info["symbols"] = [
                deepcopy(
                    {
                        "answer": answer,
                        "usage": usage,
                        "symbols": answer_list_parsed,
                    }
                )
            ]
        else:
            self.info["symbols"] += [
                deepcopy(
                    {
                        "answer": answer,
                        "usage": usage,
                        "symbols": answer_list_parsed,
                    }
                )
            ]
        return answer_list_parsed

    @property
    def priors_prompt(self):
        """Return the priors prompt for the current state."""
        if self.priors_template is None:
            raise ValueError(
                "Cannot generate priors prompt because priors template is None."
            )
        current_state = {
            "catalyst_type": self.catalyst_label,
            "inclusion_criteria": self.include_list,
            "exclusion_criteria": self.exclude_list,
            "relationship_to_candidate_list": self.relation_to_candidate_list,
        }
        actions_keys = list(current_state.keys())
        actions_descriptions = [
            "change the type of catalyst to search for",
            "add a new inclusion criteria ",
            "add a new exclusion criteria",
            "change the relationship to the candidate list",
        ]
        template_entries = {
            "current_state": convert_to_string(current_state),
            "actions_keys": convert_to_string(actions_keys, indent=0),
            "action_space": convert_to_string(actions_descriptions),
        }
        template_entries.update({"root_prompt": self.root_prompt})
        guidelines = [
            "Your catalyst type may be a category similar to, different from, or be a "
            f"subclass of {self.catalyst_label}",
            "Your new category, inclusion criteria, exclusion criteria, and "
            "relationship should not contradict those in the current $search_state.",
        ]
        if self.generation_prompt != self.root_prompt:
            current_p_a_condition = (
                f"$current_prompt = {self.generation_prompt}"
                f"\n\n$current_answer = {self.answer}"
            )
            current_conditions = (
                "$search_state, $root_prompt, $current_question and $current_answer"
            )
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": current_p_a_condition,
                    "current_conditions": current_conditions,
                }
            )
            guidelines.append(
                "Your suggestions should use scientific explanations from the answers "
                "and explanations in $current_answer"
            )
        else:
            current_conditions = "$search_state and $root_prompt"
            template_entries.update(
                {
                    "root_prompt": self.root_prompt,
                    "current_prompt_answer": "",
                    "current_conditions": current_conditions,
                }
            )
        guidelines += [
            "Your suggestions should not include MOFs, Zeolites, non-metals",
            "Your suggestions should not repeat categories from $search_state",
        ]
        guidelines_list = "\n".join([f"{i}) {g}" for i, g in enumerate(guidelines)])
        guidelines_string = (
            "Your answers should use the following guidelines:\n" + guidelines_list
        )
        template_entries.update({"guidelines": guidelines_string})
        keys_string = ", ".join(['"' + k + '"' for k in list(current_state.keys())])
        template_entries.update(
            {
                "final_task": "Let's think step-by-step, explain your "
                "thought process, with scientific justifications, then return your "
                "answer as a dictionary mapping from "
                f"[{keys_string}] "
                "to lists of suggestions."
            }
        )
        prompt = fstr(self.priors_template, template_entries)
        return prompt

    def process_prior(self, results):
        """Process the results of the prior prompt."""
        if isinstance(results, str):
            prior_answer = results
            usage = None
        else:
            prior_answer = results["answer"]
            usage = results["usage"].get("usage", None)

        action_lists = {}
        for line in prior_answer.split("{")[-1].split("\n"):
            if ":" in line:
                action, possible_actions = line.split(":")
                action_list = list(
                    {
                        s.strip().replace("'", "").replace('"', "").strip()
                        for s in possible_actions.strip()
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")
                        if s.strip().replace("'", "").replace('"', "").strip() != ""
                    }
                )  # Use a set for unique elements only
                action_lists[action.strip().strip('"')] = action_list
        if "priors" not in self.info:
            self.info["priors"] = [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_lists,
                    }
                )
            ]
        else:
            self.info["priors"] += [
                deepcopy(
                    {
                        "prompt": self.priors_prompt,
                        "answer": prior_answer,
                        "usage": usage,
                        "parsed_actions": action_lists,
                    }
                ),
            ]
        return action_lists

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
        candidate_list_statement += f"The list that you return should probably should not have the same {catalyst_label} as this list! "
        candidate_list_statement += f"Your list of {catalyst_label} may be {relation_to_candidate_list} those in the list. "
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
    ret_val = eval(f"""f'''{fstring_text}'''""", vals)
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


def convert_to_string(obj: object, indent=1):
    """Convert the given dictionary to a string for prompts."""
    if isinstance(obj, dict):
        new_dict = obj.copy()
        for k, v in obj.items():
            new_dict[k] = convert_to_string(v, indent=indent + 1)
        return json.dumps(new_dict, indent=indent)
    elif isinstance(obj, list):
        new_list = obj.copy()
        for i, v in enumerate(new_list):
            new_list[i] = convert_to_string(v, indent=indent + 1)
        return json.dumps(new_list, indent=indent)
    else:
        return str(obj)


if __name__ == "__main__":
    import pickle
    import sys

    sys.path.append("src")
    from llm.automate_prompts import get_initial_state_oc

    state, policy = get_initial_state_oc("*OHCH3", "gpt-3.5-turbo", "gpt-3.5-turbo")

    answers = [
        None,
        """To generate a list of top-5 catalysts for the adsorption of *OHCH3 with high selectivity, I considered catalysts that are known to have strong interactions with the adsorbate and facilitate its adsorption. Here are the top-5 catalysts along with scientific explanations:

Platinum (Pt): Platinum is an excellent catalyst for the adsorption of *OHCH3 due to its high reactivity and strong affinity towards oxygen-containing species. It can facilitate the binding of *OHCH3 by forming strong Pt-O bonds, which enhances the selectivity of the reaction.

Palladium (Pd): Palladium is another effective catalyst for the adsorption of *OHCH3. It has a similar reactivity to platinum and can form strong Pd-O bonds with the *OHCH3 adsorbate. The ability of palladium to stabilize the adsorbate enhances the selectivity of the reaction.

Gold (Au): Gold is known for its unique catalytic properties, including a strong affinity towards oxygen-containing species. It can interact with *OHCH3 through Au-O bonds, leading to enhanced selectivity in the adsorption process. Gold nanoparticles, in particular, have been proven to exhibit high selectivity for such reactions.

Silver (Ag): Silver is a promising catalyst for the adsorption of *OHCH3 due to its strong interaction with oxygen-containing species. The formation of Ag-O bonds with the adsorbate stabilizes the molecule, allowing for highly selective adsorption processes.

Ruthenium (Ru): Ruthenium is a transition metal that has been found to catalyze various reactions involving oxygen-containing species. It can interact with *OHCH3 through Ru-O bonds, facilitating the adsorption process and increasing the selectivity of the reaction.

Here is the final list as a Python list named "final_answer":

final_answer = ["Platinum (Pt)", "Palladium (Pd)", "Gold (Au)", "Silver (Ag)", "Ruthenium (Ru)"]

Please note that the final selection of catalysts may depend on specific reaction conditions and other factors.""",
        """Based on the list you provided, I have compared it with my recommendations for catalysts. Here are some additional catalysts that you may consider, which have chemically accurate descriptions of their interactions with adsorbates:

Rhodium (Rh): Rhodium is a transition metal catalyst that exhibits excellent catalytic properties due to its ability to form stable bonds with reactants. It can undergo various interactions with adsorbates, such as weak bonding through van der Waals forces, coordination bonding through dative covalent bonds, or even electron transfer through redox reactions.

Iron (Fe): Iron-based catalysts are widely used in various catalytic reactions. Iron can interact with adsorbates by forming coordination bonds, where the adsorbate species can coordinate with the metal center through available lone pair electrons or π-electrons.

Copper (Cu): Copper catalysts are known for their versatility and effectiveness in various catalytic processes. Copper can interact with adsorbates through ion-dipole interactions, coordination bonding, or even transfer of electrons.

Nickel (Ni): Nickel is a transition metal catalyst that can interact with adsorbates through a variety of mechanisms. It can form coordination complexes, adsorb species through weak van der Waals interactions, or even participate in electron transfer processes.

Cobalt (Co): Cobalt-based catalysts have shown excellent catalytic performance in various reactions. Cobalt can interact with adsorbates by forming coordination bonds, stabilizing adsorbed species through its available d-orbitals.

It is important to note that the recommended catalysts may interact differently with specific reactants or adsorbates, and the choice of catalyst should align with the desired reaction and reaction conditions.
final_answer = ["Rhodium (Rh)", "Iron (Fe)", "Copper (Cu)", "Nickel (Ni)", "Cobalt (Co)"]""",
        """Based on the given criteria, here are the top-5 catalysts for the adsorption of *OHCH3, excluding the catalysts from the initial list:

Palladium (Pd): Palladium is a highly selective catalyst for *OHCH3 adsorption due to its ability to form stable complexes with oxygen-containing species. It can adsorb *OHCH3 through the dissociation of the O-H bond on its surface, forming a Pd-O species.

Platinum (Pt): Platinum is an excellent catalyst for *OHCH3 adsorption due to its high selectivity and stability. It can interact with *OHCH3 by breaking the O-H bond, resulting in the formation of a Pt-O species.

Silver (Ag): Silver is a promising catalyst for *OHCH3 adsorption because it can promote the dissociation of the O-H bond on its surface, leading to the formation of an Ag-O species. Silver exhibits high selectivity and low toxicity, making it an attractive option.

Ruthenium (Ru): Ruthenium is known for its strong interaction with oxygen-containing species, making it an effective catalyst for *OHCH3 adsorption. It can adsorb *OHCH3 by breaking the O-H bond, forming a Ru-O species. Ruthenium catalysts are highly selective and generally safe for use.

Titanium dioxide (TiO2): TiO2 is a non-toxic catalyst with high selectivity for *OHCH3 adsorption. It can form stable bonds with *OHCH3 through surface hydroxide species, resulting in the formation of a Ti-O species. TiO2 catalysts are widely used and considered environmentally friendly.

Final Answer: ['Palladium (Pd)', 'Platinum (Pt)', 'Silver (Ag)', 'Ruthenium (Ru)', 'Titanium dioxide (TiO2)']""",
        """Based on the provided list, I recommend the following top-5 catalysts for the adsorption of *OHCH3, prioritizing high selectivity and excluding catalysts with high toxicity:

Gold (Au): Gold is an excellent choice for adsorbing *OHCH3 due to its high selectivity and low toxicity. The interaction between *OHCH3 and gold involves a weak chemisorption bond, facilitated by the polarizability of gold atoms. This interaction occurs through the donation of electron density from the lone pairs of oxygen atoms in *OHCH3 to vacant d-orbitals of gold atoms.

Zinc oxide (ZnO): ZnO is a promising catalyst for *OHCH3 adsorption due to its high selectivity and low toxicity. ZnO surface can interact with *OHCH3 via hydrogen bonding, where the oxygen atom of *OHCH3 forms a hydrogen bond with the oxygen atom of ZnO. This interaction enhances the adsorption capacity and selectivity towards *OHCH3.

Copper (Cu): Copper is another suitable catalyst that exhibits high selectivity and low toxicity for *OHCH3 adsorption. The interaction between *OHCH3 and copper primarily involves weak van der Waals forces and dipole-dipole attractions. These interactions are facilitated by the partial positive charge on Cu and the partial negative charge on *OHCH3, promoting strong adsorption.

Iron oxide (Fe2O3): Iron oxide is an effective catalyst with high selectivity and low toxicity for the adsorption of *OHCH3. The adsorption process involves the formation of coordination bonds between the oxygen atoms of *OHCH3 and the surface iron(III) ions. This interaction enhances the adsorption capacity and selectivity towards *OHCH3.

Manganese dioxide (MnO2): MnO2 is a desirable catalyst with high selectivity and low toxicity for *OHCH3 adsorption. The adsorption mechanism involves the formation of hydrogen bonds between *OHCH3 and manganese dioxide, where the oxygen atom of *OHCH3 interacts with the oxygen atom of MnO2. This interaction strengthens the adsorption and improves selectivity.

Final answer: ['Gold (Au)', 'Zinc oxide (ZnO)', 'Copper (Cu)', 'Iron oxide (Fe2O3)', 'Manganese dioxide (MnO2)']""",
        """To generate a list of top-5 catalysts for the adsorption of *OHCH3 with high selectivity, excluding catalysts with high toxicity and low activity, we will evaluate the suitability of each candidate catalyst. Here are scientific explanations for each of the catalysts:

Palladium (Pd): Palladium is a widely used catalyst for various organic transformations due to its high catalytic activity and selectivity. It can adsorb *OHCH3 effectively on its surface, allowing for efficient reaction rates. Palladium catalysts are generally nontoxic, making them suitable for use in catalytic processes.

Ruthenium (Ru): Ruthenium catalysts have shown excellent selectivity in a wide range of catalytic reactions. They can adsorb *OHCH3 on their surface and facilitate the desired reaction with high efficiency. Ruthenium is also known for its low toxicity and high stability, making it a suitable catalyst for various applications.

Platinum (Pt): Platinum catalysts possess both excellent activity and selectivity in numerous catalytic processes. Pt can efficiently adsorb *OHCH3 and enhance the desired reaction pathways. Platinum is also considered a safe catalyst with low toxicity and high resistance to poisoning, making it a favorable choice for catalytic applications.

Rhodium (Rh): Rhodium catalysts have been proven to exhibit remarkable selectivity and stability in various catalytic reactions. Rh can effectively adsorb *OHCH3 and promote the desired reaction with high activity. Rhodium is generally nontoxic and resistant to deactivation, making it a suitable choice for catalysis.

Titanium dioxide (TiO2): Titanium dioxide catalysts are widely used due to their excellent selectivity and stability. TiO2 can adsorb *OHCH3 efficiently, facilitating the desired reaction with high activity. Titanium dioxide is considered safe and non-toxic, making it a suitable catalyst for various applications.

Now let's generate the final list:

final_answer = ['Palladium (Pd)', 'Ruthenium (Ru)', 'Platinum (Pt)', 'Rhodium (Rh)', 'Titanium dioxide (TiO2)']

Please note that these catalysts have been selected based on their properties of high selectivity, low toxicity, and high activity, making them suitable for the adsorption of *OHCH3.""",
        """To generate a list of top-5 catalysts for the adsorption of *OHCH3 with high selectivity and activity, we will first review the properties of the catalysts in the initial list: ['Palladium (Pd)', 'Ruthenium (Ru)', 'Platinum (Pt)', 'Rhodium (Rh)', 'Titanium dioxide (TiO2)'].

Palladium (Pd): Palladium is a highly efficient catalyst for various reactions due to its ability to adsorb reactants on its surface and facilitate their conversion. For the adsorption of *OHCH3, we can consider a Pd catalyst supported on titanium(IV) oxide (Pd/TiO2). The Pd surface acts as the active site where the *OHCH3 adsorbs, while the role of TiO2 is to provide stability and increase the selectivity by preventing side reactions.

Ruthenium (Ru): Ruthenium is known for its catalytic activity and stability. For our purpose, we can consider a Ru catalyst supported on carbon (Ru/C). The Ru surface provides active sites for *OHCH3 adsorption, while the carbon support enhances the activity and selectivity through its interaction with the reactants and the stabilization of the catalyst.

Platinum (Pt): Platinum is a versatile catalyst widely used in organic synthesis. In this case, we can recommend a Pt catalyst supported on alumina (Pt/Al2O3). The Pt surface acts as the active site for the *OHCH3 adsorption, while the alumina support enhances the selectivity by providing a stable environment and promoting specific interactions with the adsorbate.

Rhodium (Rh): Rhodium is highly active and exhibits excellent catalytic properties. Considering the adsorption of *OHCH3, we can suggest a Rh catalyst supported on silica (Rh/SiO2). The Rh surface provides active sites for *OHCH3 adsorption, while the silica support contributes to high selectivity by enhancing adsorbate-substrate interactions and preventing catalyst deactivation.

Zirconium dioxide (ZrO2): To expand our options further, we can consider a ZrO2 catalyst. Zirconium dioxide has shown promising catalytic properties in various reactions. In this case, we can recommend a ZrO2 catalyst modified with platinum (Pt/ZrO2). The ZrO2 surface provides potential adsorption sites, and the modification with Pt enhances activity and selectivity through synergistic effects and improved reactant-substrate interactions.

With the consideration of these catalysts, we can generate the final list of top-5 catalysts as follows:

final_answer = ['Palladium supported on titanium dioxide (Pd/TiO2)', 'Ruthenium supported on carbon (Ru/C)', 'Platinum supported on alumina (Pt/Al2O3)', 'Rhodium supported on silica (Rh/SiO2)', 'Platinum modified zirconia (Pt/ZrO2)']

Please note that the final_answer list includes chemically accurate descriptions of the catalysts and their interaction with the adsorbate.""",
    ]
    states = [state]
    action_idxs = [1, 19, 14, 19, 9, 0]

    for i in range(1, 7):
        actions, priors = policy.get_actions([state])
        priors = priors[0]
        if len(action_idxs) <= i - 1:
            action_idx = None
            while action_idx is None or priors[action_idx] == 0.0:
                action_idx = np.random.randint(len(actions))
                print(action_idx)
        else:
            action_idx = action_idxs[i - 1]
            print(action_idx)

        state = actions[action_idx](state)
        print(actions[action_idx])
        print(state.generation_system_prompt)
        print("-" * 15)
        print(state.generation_prompt)

        llm_function = lambda a: answers[i]  # noqa: E731
        state.process_generation(llm_function(state))
        states.append(state)

        print(llm_function(state))

        print(("*" * 15 + "\n") * 4)
