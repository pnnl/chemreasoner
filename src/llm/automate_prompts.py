"""Functions to automate prompts for the bio fuels dataset."""
import sys

import pandas as pd

sys.path.append("src")
from llm import query  # noqa: E402
from search.policy.reasoner_policy import ReasonerPolicy  # noqa: E402


def find_all(string, sub):
    """Find all instances of sub string in a string."""
    start = 0
    while True:
        start = string.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_initial_state_oc(
    adsorbate: str, prediction_model, reward_model, simulation_reward=False
):
    """Get initial state for LLM query from adsorbate string."""
    template = (
        "Generate a list of top-5 {catalyst_label} "
        f"for the adsorption of {adsorbate}."
        "{include_statement}{exclude_statement}"
        "Provide scientific explanations for each of the catalysts. "
        "Finally, return a python list named final_answer which contains the top-5 catalysts. "
        "{candidate_list_statement}"
        r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
    )
    starting_state = query.QueryState(
        template=template,
        reward_template=None,
        ads_symbols=[adsorbate],
        relation_to_candidate_list="include elements similar to",
        ads_preferences=[1],
        num_answers=5,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    if simulation_reward:
        policy = ReasonerPolicy(
            catalyst_label_types=["", "monometallic ", "bimetallic ", "trimetallic "],
            try_oxides=False,
        )
    else:
        policy = ReasonerPolicy()
    return starting_state, policy


def non_rwgs_template_generator(
    question: str, prediction_model, reward_model, simulation_reward=False
):
    """Generate initial query for non RWGS reaction prompt."""
    adsorbate = question.split("bind ")[1].split(" in")[0]
    reaction_name = question.split("in ")[1].split(" reaction")[0]
    property_name = question.split("with ")[1].split(".")[0].lower()

    template = (
        "What are the top-3 {catalyst_label} that"
        + "perform the "
        + f"{reaction_name} reaction and demonstrate higher adsorption energy for"
        + f"{adsorbate}?. "
        + "{include_statement}{exclude_statement}"
        + "Provide scientific explanations for each of the catalysts. "
        + "Finally, return a python list named final_answer which contains the top-5 catalysts. "
        "{candidate_list_statement}"
        r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
    )

    qs = query.QueryState(
        template=template,
        reward_template=None,
        ads_symbols=[adsorbate],
        num_answers=3,
        include_list=[property_name],
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    if simulation_reward:
        this_policy = ReasonerPolicy(
            catalyst_label_types=["", "monometallic ", "bimetallic ", "trimetallic "],
            try_oxides=False,
        )
    else:
        this_policy = ReasonerPolicy()
    return qs, this_policy


def parse_rwgs_questions(
    question: str, prediction_model, reward_model, simulation_reward=False
):
    """Parse the rwgs reaction questions."""
    catalyst_type, cheap_statement = parse_parameters_from_question(question)
    if catalyst_type is not None:
        if catalyst_type != " catalysts":
            catalyst_label_types = [catalyst_type[1:-1]]
        else:
            if simulation_reward:
                catalyst_label_types = [
                    "",
                    "monometallic ",
                    "bimetallic ",
                    "trimetallic ",
                ]
            else:
                catalyst_label_types = None
        question = question.replace(catalyst_type, "{catalyst_label}")  # Remove {}
    else:
        catalyst_label_types = None

    if cheap_statement is not None:
        if "cheap" in cheap_statement:
            include_list = ["low cost"]
        else:
            raise ValueError(f"Unkown value {cheap_statement}")
        question = question.replace(cheap_statement, "")
    else:
        include_list = []

    ads_symbols = []
    ads_preference = []
    for possible_ads in ["CO", "CO2", "H2"]:
        if possible_ads in question:
            if possible_ads == "CO":
                all_co = list(find_all(question, "CO"))
                all_co2 = list(find_all(question, "CO2"))
                if all_co == all_co2:
                    continue
            ads_symbols.append(possible_ads)
            preference = -1 if possible_ads == "CO" else 1
            ads_preference.append(preference)

    # If there's no adsorbates in the prompt...
    if len(ads_symbols) == 0:
        ads_symbols = ["CO", "CO2", "H2"]
        ads_preference = [-1, 1, 1]

    question = question.replace("top", "top-3")
    template = (
        f"{question} "
        "{include_statement}{exclude_statement}"
        "Provide scientific explanations for each of the catalysts. "
        "Finally, return a python list named final_answer which contains the top-3 catalysts. "
        "{candidate_list_statement}"
        r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
    )
    qs = query.QueryState(
        template=template,
        reward_template=None,
        ads_symbols=ads_symbols,
        ads_preferences=ads_preference,
        include_list=include_list,
        num_answers=3,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    if simulation_reward:
        this_policy = ReasonerPolicy(
            catalyst_label_types=catalyst_label_types, try_oxides=False
        )
    else:
        this_policy = ReasonerPolicy(
            catalyst_label_types=catalyst_label_types,
        )
    return qs, this_policy


def parse_parameters_from_question(question: str):
    """Parse out parameters, such as "cheap" and catalyst type from a question."""
    if "{" in question and "}" in question:
        catalyst_type = question[question.find("{") : question.find("}") + 1]  # noqa
    else:
        catalyst_type = None

    if "[" in question and "]" in question:
        cheap_statement = question[question.find("[") : question.find("]") + 1]  # noqa
    else:
        cheap_statement = None
    return (catalyst_type, cheap_statement)


def get_initial_state_biofuels(
    row, prediction_model, reward_model, simulation_reward=False
):
    """Get query state and policy from a row."""
    print("\n--------------\n")
    if row["Question"] in ["", "Non-RWGS", "RWGS"] or pd.isna(row)["Question"]:
        state_policy = None
    elif any(
        rxn_name in row["Question"]
        for rxn_name in ["hydrogenation", "hydrodeoxygenation"]
    ):
        print("clean_question")
        state_policy = non_rwgs_template_generator(
            row["Question"],
            prediction_model,
            reward_model,
            simulation_reward=simulation_reward,
        )
    elif "crystal planes" in row["Question"]:
        print(f"SKIPPING: {row['Question']}")
        state_policy = None
    elif any(
        [bad_string in row["Question"] for bad_string in ["structure", "particle"]]
    ):
        print(f"SKIPPING: {row['Question']}")
        state_policy = None
    else:
        state_policy = parse_rwgs_questions(
            row["Question"],
            prediction_model,
            reward_model,
            simulation_reward=simulation_reward,
        )

    return state_policy


if __name__ == "__main__":
    df = pd.read_csv("src/reasoner/Answer2Questions.csv")
    unautomated_counter = 0
    omitted_row_counter = 0
    for i, row in df.iterrows():
        print("\n--------------\n")
        if row["Question"] in ["", "Non-RWGS", "RWGS"] or pd.isna(row)["Question"]:
            omitted_row_counter += 1
        elif any(
            rxn_name in row["Question"]
            for rxn_name in ["hydrogenation", "hydrodeoxygenation"]
        ):
            print("clean_question")
            s, p = non_rwgs_template_generator(row["Question"])
        elif "crystal planes" in row["Question"]:
            print(f"SKIPPING: {row['Question']}")
            unautomated_counter += 1
        elif any(
            [bad_string in row["Question"] for bad_string in ["structure", "particle"]]
        ):
            print(f"SKIPPING: {row['Question']}")
            unautomated_counter += 1
        else:
            s, p = parse_rwgs_questions(row["Question"])
            print(s.ads_symbols)
            print(s.ads_preferences)
            unautomated_counter += 1

    num_total = len(df) - omitted_row_counter
    num_automated = num_total - unautomated_counter
    print(f"{num_automated}/{num_total} = {num_automated/num_total*100}%")
