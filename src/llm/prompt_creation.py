from ast import literal_eval


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


# Functions you need #


def parse_answer(answer: str, num_expected=None):
    """Parse an answer into a list of catalysts."""
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


def energy_calculation_prompt(adsorbates, catalyst_list):
    """Reurn a list of allt he energy calculation prompts."""
    prompts = []
    for ads in adsorbates:
        prompts.append(
            generate_adsorption_energy_list_prompt(
                adsorbate=ads, candidate_list=catalyst_list
            )
        )
    return prompts


def parsing_prompt(candidate_list: list[str]):
    """Turn an llm_answer into a prompt for symbols_parsing."""

    example_format = ""
    for i, ans in enumerate(candidate_list):
        example_format += f"{ans}: [list_{i}]\n"

    answer_string = ", ".join(candidate_list)
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


if __name__ == "__main__":
    import json

    with open("json_database.json", "r") as f:
        data = json.load(f)

    for d in data:

        print(
            energy_calculation_prompt(
                d["energy_calculation_prompt"]["adsorbates"], ["Pt", "Pd", "Ru"]
            )
        )

        print(parsing_prompt(["Pt", "Pd", "Ru"]))
