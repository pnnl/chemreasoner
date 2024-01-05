"""Functions to automate prompts for the bio fuels dataset."""
import sys

import pandas as pd

sys.path.append("src")
from search.state.reasoner_state import ReasonerState  # noqa: E402

molecule_conversions = {
    "CO2": "CO2",
    "CO": "*CO",
    "H2O": "*OH2",
    "methanol": "*OHCH3",
    "ethanol": "*OHCH2CH3",
}


def find_all(string, sub):
    """Find all instances of sub string in a string."""
    start = 0
    while True:
        start = string.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_template(question, chain_of_thought):
    """Get the template for the given quesiton."""
    template = question
    if chain_of_thought:
        template += (
            "{include_statement} {exclude_statement}"
            "Provide scientific explanations for each of the catalysts. "
            "Finally, return a python list named final_answer which contains the top-5 catalysts. "
            "{candidate_list_statement}"
            r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
        )
    else:
        template += (
            "{include_statement} {exclude_statement}"
            "{candidate_list_statement}"
            r"\n\nReturn a python list named final_answer which contains the top-5 catalysts."
        )
    return template


def get_initial_state_open_catalyst(
    question,
    prediction_model,
    reward_model,
    simulation_reward=False,
    chain_of_thought=True,
):
    """Get initial state for LLM query from adsorbate string."""
    template = question.replace("{catalysts}", "{catalyst_label}")
    if chain_of_thought:
        template += (
            "{include_statement} {exclude_statement}"
            "Provide scientific explanations for each of the catalysts. "
            "Finally, return a python list named final_answer which contains the top-5 catalysts. "
            "{candidate_list_statement}"
            r"\n\nTake a deep breath and let's think step-by-step. Remember, you need to return a python list named final_answer!"
        )
    else:
        template += (
            "{include_statement} {exclude_statement}"
            "{candidate_list_statement}"
            r"\n\nReturn a python list named final_answer which contains the top-5 catalysts."
        )
    adsorbate = question.split("adsorption of ")[-1].split(".")[0]
    starting_state = ReasonerState(
        template=template,
        reward_template=None,
        ads_symbols=[adsorbate],
        ads_preferences=[1],
        num_answers=5,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    return starting_state


def get_initial_state_bio_fuels(
    question: str,
    prediction_model,
    reward_model,
    simulation_reward=False,
    chain_of_thought=True,
):
    """Generate initial query for non RWGS reaction prompt."""
    adsorbate = question.split("bind ")[1].split(" in")[0]
    reaction_name = question.split("in ")[1].split(" reaction")[0]
    property_name = question.split("with ")[1].split(".")[0].lower()
    template = get_template(question)

    qs = ReasonerState(
        template=template,
        reward_template=None,
        ads_symbols=[adsorbate],
        num_answers=3,
        include_list=[property_name],
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    return qs


def get_initial_state_rwgs(
    question: str,
    prediction_model,
    reward_model,
    simulation_reward=False,
    chain_of_thought=True,
):
    """Parse the rwgs reaction questions."""
    catalyst_type, cheap_statement = parse_parameters_from_question(question)
    question = question.replace(catalyst_type, "{catalyst_label}")

    if cheap_statement is not None:
        if "cheap" in cheap_statement:
            include_list = ["low cost"]
        else:
            raise ValueError(f"Unkown value {cheap_statement}")

    else:
        include_list = []
    template = get_template(question)

    ads_symbols = []
    ads_preference = []
    for possible_ads in ["CO", "CO2", "H2"]:
        if possible_ads in question.replace("RWGS reaction", ""):
            ads_symbols.append(molecule_conversions[possible_ads])
            preference = -1 if possible_ads == "CO" else 1
            ads_preference.append(preference)

    # If there are no adsorbates in the prompt...
    if len(ads_symbols) == 0:
        # Do the reaction
        ...

    qs = ReasonerState(
        template=template,
        reward_template=None,
        ads_symbols=ads_symbols,
        ads_preferences=ads_preference,
        include_list=include_list,
        num_answers=3,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    return qs


def get_initial_state_methanol(
    question: str,
    prediction_model,
    reward_model,
    simulation_reward=False,
    chain_of_thought=True,
):
    """Parse the rwgs reaction questions."""
    catalyst_type, cheap_statement = parse_parameters_from_question(question)
    question = question.replace(catalyst_type, "{catalyst_label}")

    if cheap_statement is not None:
        if "cheap" in cheap_statement:
            include_list = ["low cost"]
        else:
            raise ValueError(f"Unkown value {cheap_statement}")

    else:
        include_list = []
    template = get_template(question)

    ads_symbols = []
    ads_preference = []
    for possible_ads in ["methanol", "CO2", "H2"]:
        if possible_ads in question.replace("CO2 to methanol conversion reaction", ""):
            ads_symbols.append(molecule_conversions[possible_ads])
            preference = -1 if possible_ads == "methanol" else 1
            ads_preference.append(preference)

    # If there are no adsorbates in the prompt...
    if len(ads_symbols) == 0:
        # Do the reaction
        ...

    qs = ReasonerState(
        template=template,
        reward_template=None,
        ads_symbols=ads_symbols,
        ads_preferences=ads_preference,
        include_list=include_list,
        num_answers=3,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    return qs


def get_initial_state_ethanol(
    question: str,
    prediction_model,
    reward_model,
    simulation_reward=False,
    chain_of_thought=True,
):
    """Parse the rwgs reaction questions."""
    catalyst_type, cheap_statement = parse_parameters_from_question(question)
    question = question.replace(catalyst_type, "{catalyst_label}")

    if cheap_statement is not None:
        if "cheap" in cheap_statement:
            include_list = ["low cost"]
        else:
            raise ValueError(f"Unkown value {cheap_statement}")

    else:
        include_list = []
    template = get_template(question)

    ads_symbols = []
    ads_preference = []
    for possible_ads in ["methanol", "CO2", "H2"]:
        if possible_ads in question.replace("CO2 to ethanol conversion reaction", ""):
            ads_symbols.append(molecule_conversions[possible_ads])
            preference = -1 if possible_ads == "ethanol" else 1
            ads_preference.append(preference)

    # If there are no adsorbates in the prompt...
    if len(ads_symbols) == 0:
        # Do the reaction
        ...

    qs = ReasonerState(
        template=template,
        reward_template=None,
        ads_symbols=ads_symbols,
        ads_preferences=ads_preference,
        include_list=include_list,
        num_answers=3,
        prediction_model=prediction_model,
        reward_model=reward_model,
    )
    return qs


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
