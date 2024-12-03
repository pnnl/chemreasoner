import os
import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
import random  # type: ignore


def read_molecular_structure(file_path):
    structure = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = parts[1:4]
            structure.append([float(x), float(y), float(z)])
    return np.array(structure)


def calculate_distance(structure1, structure2):
    # Ensure structures have the same number of atoms
    if len(structure1) != len(structure2):
        return float("inf")

    # Calculate the pairwise distances between atoms
    distances = cdist(structure1, structure2)

    # Return the sum of minimum distances for each atom
    return np.sum(np.min(distances, axis=1))


def nearest_neighbor_sort(structures, filenames):
    n = len(structures)
    unvisited = set(range(n))
    path = []

    # Start with a random structure
    current = random.choice(list(unvisited))
    path.append(current)
    unvisited.remove(current)

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda x: calculate_distance(structures[current], structures[x]),
        )
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return [filenames[i] for i in path]


def sort_structures(directory):
    # Read all structures
    structures = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".xyz"):
            file_path = os.path.join(directory, filename)
            structures.append(read_molecular_structure(file_path))
            filenames.append(filename)

    # Sort structures using nearest neighbor heuristic
    return nearest_neighbor_sort(structures, filenames)


if __name__ == "__main__":

    import csv
    import glob
    import json

    for node in glob.glob("data/*"):
        if not os.path.isdir(node):
            continue
        with open(os.path.join(node, "reward_values.csv")) as f:
            reward_values = {
                row["id"]: float(row["reward"]) for row in csv.DictReader(f)
            }
        dirname = os.path.join(node, "relaxed_structures/trajectories_e_tot")
        sorted_structures = sort_structures(dirname)
        structures = []
        for filename in sorted_structures:
            id = os.path.basename(filename).split("_")[0]
            with open(os.path.join(dirname, filename), "r") as f:
                structures.append(
                    {
                        "id": id,
                        "reward": reward_values.get(id),
                        "structure": f.read(),
                    }
                )
        with open(os.path.join(node, "structures_with_rewards.json"), "w") as f:
            json.dump(structures, f)
    # directory = "data/cu_zn_test/relaxed_structures/trajectories_e_tot"
    # sorted_structures = sort_structures(directory)
    # print("Sorted order of structures:")
    # for filename in sorted_structures:
    #     print(filename)
