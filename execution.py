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
    """
    Run algorithm on the function.
    :param function: Function on which to run.
    :param algorithm: Algorithm to execute.
    :param population_size: Population size.
    :param generations: How many generations to execute.
    :param initialization: Initialization of the first population.
    :param show_progress: Whether to show progress.
    :return: tuple of population in shape (generations, population_size, function.dimension) and evaluations in shape (generations, population_size)
    """
    populations = []
    evaluations = []

    counter = range(generations-1)
    if show_progress:
        counter = pb(counter)

    population = initialization(function.lower_bounds, function.upper_bounds, [population_size, function.dimension])
    for gen in counter:
        populations.append(population if isinstance(population, np.ndarray) else population.numpy())
        evaluations.append(np.apply_along_axis(function, axis=1, arr=populations[-1]))
        population = algorithm(population, evaluations[-1])

    return np.stack(populations), np.stack(evaluations)

def execute_multiple(
        function: cocoex.Problem,
        algorithm: Callable,
        repeats: int = 10,
        population_size: int = 100,
        generations: int = 100,
        initialization: Callable = np.random.uniform,
        show_progress = False
) -> Tuple[np.array, np.array]:
    """
    Run algorithm multiple times on the function.
    :param function: Function on which to run.
    :param algorithm: Algorithm to execute.
    :param repeats: How many times to repeat the execution.
    :param population_size: Population size.
    :param generations: How many generations to execute.
    :param initialization: Initialization of the first population.
    :param show_progress: Whether to show progress.
    :return: tuple of population in shape (repeats, generations, population_size, function.dimension) and evaluations in shape (repeats, generations, population_size)
    :return:
    """
    counter = range(repeats)
    if show_progress:
        counter = pb(counter)

    populations = []
    evaluations = []
    for iteration in counter:
        population, evaluation = execute(function, algorithm, population_size, generations, initialization, False)
        populations.append(population)
        evaluations.append(evaluation)

    return np.stack(populations), np.stack(evaluations)

