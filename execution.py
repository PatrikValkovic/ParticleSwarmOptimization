import cocoex
import numpy as np
from typing import Callable, Tuple
from fstpso import FuzzyPSO
from progressbar import progressbar as pb

def evaluate(
        function: cocoex.Problem,
        population: np.ndarray
):
    return np.apply_along_axis(function, axis=1, arr=population)

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
    :return: Tuple of population in shape (generations, population_size, function.dimension) and evaluations in shape (generations, population_size).
    """
    populations = []
    evaluations = []

    counter = range(generations-1)
    if show_progress:
        counter = pb(counter)

    population = initialization(function.lower_bounds, function.upper_bounds, [population_size, function.dimension])
    for gen in counter:
        populations.append((population if isinstance(population, np.ndarray) else population.numpy()).copy())
        evaluations.append(evaluate(function, populations[-1]))
        population = algorithm(population, evaluations[-1], function)

    return np.stack(populations, axis=0), np.stack(evaluations, axis=0)

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
    :return: Tuple of population in shape (repeats, generations, population_size, function.dimension) and evaluations in shape (repeats, generations, population_size).
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


def fstpso(
    function: cocoex.Problem,
    population_size: int = 100,
    generations: int = 100,
    show_progress=False
):
    """
    Run FST-PSO algorithm.
    :param function: Function to optimize.
    :param population_size: Number of particles in the swarm.
    :param generations: How many iterations to perform.
    :param show_progress: Whether to show progress (verbose logging).
    :return: Tuple of population in shape (generations, population_size, function.dimension) and evaluations in shape (generations, population_size).
    """
    populations = []
    values = []

    def _callback(fpso):
        populations.append(list(map(lambda particle: particle.B, fpso.Solutions)))
        values.append(list(map(lambda particle: particle.CalculatedFitness, fpso.Solutions)))

    FPSO = FuzzyPSO()
    FPSO.set_search_space(np.stack([function.lower_bounds, function.upper_bounds], axis=1))
    FPSO.set_swarm_size(population_size)
    FPSO.set_fitness(function)
    FPSO.solve_with_fstpso(callback={
        'function': _callback,
        'interval': 1
    }, max_iter=generations-1, verbose=show_progress)
    return np.array(populations), np.array(values)

def fstpso_multiple(
        function: cocoex.Problem,
        repeats = 10,
        population_size: int = 100,
        generations: int = 100,
        show_progress=False
) -> Tuple[np.array, np.array]:
    """
    Run the FST-PSO algorithm multiple times.
    :param function: Function to optimize.
    :param repeats: How many times to repeat the execution.
    :param population_size: Number of particles in the swarm.
    :param generations: How many iterations to perform.
    :param show_progress: Whether to show progress (verbose logging).
    :return: Tuple of population in shape (repeats, generations, population_size, function.dimension) and evaluations in shape (repeats, generations, population_size).
    """
    counter = range(repeats)
    if show_progress:
        counter = pb(counter)

    populations = []
    evaluations = []
    for iteration in counter:
        population, evaluation = fstpso(function, population_size, generations, False)
        populations.append(population)
        evaluations.append(evaluation)

    return np.stack(populations), np.stack(evaluations)
