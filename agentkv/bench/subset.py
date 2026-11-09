"""Sample complete roots without depending on source row order."""

import numpy as np


def sample_roots(root_ids: list[str], seed: int, num_roots: int) -> list[str]:
    population = sorted(set(root_ids))
    if len(population) != len(root_ids):
        raise ValueError("Duplicate root IDs in the population")
    if seed < 0 or not 1 <= num_roots <= len(population):
        raise ValueError("Require seed >= 0 and 1 <= num_roots <= population size")
    rng = np.random.Generator(np.random.PCG64(seed))
    return sorted(rng.choice(population, size=num_roots, replace=False).tolist())
