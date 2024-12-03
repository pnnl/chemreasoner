import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
import random  # type: ignore
from typing import Generator, Tuple


def read_molecular_structure(lines):
    structure = []
    for line in lines:
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


def nearest_neighbor_sort(structures: list, filenames: list[str]):
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

    return [(filenames[i], structures[i]) for i in path]


def sort_structures(structs: Generator[Tuple[str, list[str]], None, None]):
    structures = []
    filenames = []
    for filename, lines in structs:
        structures.append(read_molecular_structure(lines))
        filenames.append(filename)

    # Sort structures using nearest neighbor heuristic
    return nearest_neighbor_sort(structures, filenames)
