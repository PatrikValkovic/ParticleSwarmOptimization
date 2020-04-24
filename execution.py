import cocoex
import numpy as np
from typing import Callable, Tuple
from progressbar import progressbar as pb

def execute(
        function: cocoex.Problem,
        algorithm: Callable,
        population_size: int = 100,
        generations: int = 100,
        initialization: Callable = np.random.uniform,
        show_progress = False
) -> Tuple[np.array, np.array]:
    populations = []
    values = []

    counter = range(generations-1)
    if show_progress:
        counter = pb(counter)

    populations.append(initialization(function.lower_bounds, function.upper_bounds, [population_size, function.dimension]))
    values.append(np.apply_along_axis(function, axis=1, arr=populations[-1]))
    for gen in counter:
        new_population = algorithm(populations[-1], values[-1])
        new_population = np.maximum(function.lower_bounds, np.minimum(function.upper_bounds, new_population))
        populations.append(new_population)
        values.append(np.apply_along_axis(function, axis=1, arr=populations[-1]))

    return np.stack(populations), np.stack(values)
