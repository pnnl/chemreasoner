"""Figure out what is in tha language model output."""
import math

from ase.data import atomic_numbers, atomic_names, atomic_masses, chemical_symbols

import pandas as pd

from elements import elements

df = pd.read_csv("data/llm_answers.csv")

print(df.columns[:12])

print(df[df.columns[:12]])


def get_element(element_string: str):
    return getattr(elements, element_string)


def is_mof(answer):
    indicator = "MOF"
    if "mof" in answer.lower():
        return indicator
    elif all([s in answer for s in ["metal", "organic", "framework"]]):
        return indicator
    else:
        return None


def is_zeolite(answer):
    indicator = "Zeolite"
    if "zeoli" in answer.lower():
        return indicator


def is_nanotube(answer):
    indicator = "Nanotube"
    if "nanotube" in answer.lower():
        return indicator
    if "CNT" in answer:
        return indicator


def is_clay(answer):
    inidcator = "Clay"
    if "clay" in answer.lower():
        return inidcator


def is_ion_exchange(answer):
    indicator = "Ionic Other"
    if "ion" in answer.lower():
        return indicator


def is_acid(answer):
    indicator = "Acid"
    if "acid" in answer.lower():
        return indicator
    elif "HCl" in answer:
        return indicator
    elif "boric" in answer.lower():
        return indicator


def is_raney(answer):
    indicator = "Raney"
    if "aney" in answer.lower():
        return indicator


nonmetal_additional_names = [
    "boride",
    "carbide",
    "silica",
    "sulfide",
    "nitride",
    "nitride",
    "Flouride",
    "chloride",
    "bromide",
]

oxide_additional_names = [
    "borate",
    "carbonate",
    "oxide",
    "sulfate",
    "sulfite",
    "nitrate",
    "bromate",
    "silicate",
    "chromite",
]


def has_atomic_names(answer):
    hits = [n for n in atomic_names[1:] if n.lower() in answer.lower()]
    if len(hits) > 0:
        if "alumina" in answer.lower():
            return "Oxide"
        elif any([n in answer.lower() for n in oxide_additional_names]):
            classes = []
            for h in hits:
                h = "aluminum" if h.lower() == "aluminium" else h
                classes.append(get_element(h.lower()).Classification.lower())
            return "Oxide"
        else:
            classes = []
            for h in hits:
                h = "aluminum" if h.lower() == "aluminium" else h
                classes.append(get_element(h.lower()).Classification.lower())
            if not any(
                [c in classes for c in ["halogen", "rare_gases", "non-metals"]]
            ) and not any(
                [any([n in answer.lower() for n in nonmetal_additional_names])]
            ):
                if len(hits) == 1:
                    print(answer)
                    return "Metal"
                elif len(hits) > 1:
                    return "Metal Alloy"
            else:
                return "Metal and Non-Metal"
    return None


def has_atomic_symbols(answer):
    hits = [n for n in chemical_symbols[1:] if n in answer]
    if len(hits) > 0:
        # filter substring mathces
        hits = reversed(sorted(hits, key=len))
        copy_answer = answer
        true_hits = []
        for h in hits:
            if h in copy_answer:
                copy_answer = copy_answer.replace(h, "")
                true_hits.append(h)

        hits = true_hits

        if "O" in hits:
            classes = []
            for h in hits:
                classes.append(get_element(h).Classification.lower())
            return "Oxide"
        else:
            classes = []
            for h in hits:
                classes.append(get_element(h).Classification.lower())
            if not any(
                [
                    c in classes
                    for c in ["halogen", "rare_gases", "non-metals", "metalloid"]
                ]
            ):
                if len(hits) == 1:
                    return "Metal"
                elif len(hits) > 1:
                    return "Metal Alloy"
            else:
                return "Metal and Non-Metal"

        return None


def parse_answer(answer_string: str):
    """Parse the answer out."""
    functions = [
        is_mof,
        is_zeolite,
        is_nanotube,
        is_clay,
        is_acid,
        is_ion_exchange,
        is_raney,
        has_atomic_names,
        has_atomic_symbols,
    ]
    for f in functions:
        if f(answer_string) is not None:
            return f(answer_string)
    return None


classes_found = {}

hits = 0
count = 0
for i, r in df[df.columns[:12]].iterrows():
    for elem in r:
        if isinstance(elem, str):
            cat = parse_answer(elem)
            if cat is not None and cat in classes_found.keys():
                classes_found[cat] += 1
            elif cat is not None and cat not in classes_found.keys():
                classes_found[cat] = 1
            if cat == "Metal and Non-Metal":
                print(elem)
            hits += 1 if cat is not None else 0
            count += 1
classes_found["Uncategorized"] = (count - hits) / count * 100

print(hits / count * 100)
print(dir(elements))
print(get_element("Si").Classification)


print("---")
clean_stats = {"Other": 0}
for k, v in classes_found.items():
    if k in ["Ionic Other", "Raney", "Clay", "Acid", "Nanotube"]:
        clean_stats["Other"] += v
    else:
        clean_stats[k] = v

for c, v in clean_stats.items():
    print(f"{c}:\t{v/count*100}")
