import cocoex
import execution as exe
import numpy as np
import utils


def numeric_gradient_descent(
        particles: np.ndarray,
        values: np.ndarray,
        function: cocoex.Problem,
        gradient_step: float,
        computation_step: float) -> np.ndarray:
    gradients = utils.compute_gradient(function, particles, gradient_step)
    new_particles = particles - gradients * computation_step
    return np.maximum(-5, np.minimum(5, new_particles))


class RandomWalkObject:
    def __init__(self, lowerbound: np.ndarray, upperbound: np.ndarray, shape):
        self.population = np.random.uniform(lowerbound, upperbound, shape)
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def numpy(self):
        return self.population

    def execute(self, values: np.ndarray, function: cocoex.Problem):
        self.population = self.population + np.random.normal(0, 0.1, self.population.shape)
        self.population = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.population))
        return self

    @staticmethod
    def initialize(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return RandomWalkObject(lowerbound, upperbound, shape)

def EVAdifferencal(population: np.ndarray, fitnesses: np.ndarray, function:cocoex.Problem,
                   parents_fraction: float = 0.7,
                   F: float = 0.8,
                   CR: float = 0.4,
                   ) -> np.ndarray:
    num_parents = int(len(population) * parents_fraction)
    parents_tournament_indices = np.random.randint(0, len(population), [2, num_parents])
    comparison = fitnesses[parents_tournament_indices[0]] < fitnesses[parents_tournament_indices[1]]
    better = np.concatenate([
        parents_tournament_indices[0, comparison],
        parents_tournament_indices[1, np.logical_not(comparison), ]
    ])
    parents = population[better]

    num_children = len(population) - num_parents
    picked_parents = np.random.randint(0, num_parents, [4, num_children])
    crossover_sample = np.random.random([num_children, function.dimension]) > CR
    mutated = parents[picked_parents[0], :] + F * (parents[picked_parents[1]] - parents[picked_parents[2]])
    mutated[crossover_sample] = parents[picked_parents[3]][crossover_sample]
    mutated = np.maximum(-5, np.minimum(5, mutated))

    return np.concatenate([
        parents,
        mutated
    ])


class Standard2006:
    def __init__(self, lowerbound: np.ndarray, upperbound: np.ndarray, population, dimension):
        self.population = np.random.uniform(lowerbound, upperbound, [population, dimension])
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.initialized = False
        self.velocity = None
        self.best_pos = None
        self.best_val = None
        self.bestneigh_pos = None
        self.bestneigh_val = None

    def numpy(self):
        return self.population

    def execute(self, values: np.ndarray, function: cocoex.Problem):
        if not self.initialized:
            self.velocity = (np.random.uniform(self.lowerbound, self.upperbound, self.population.shape) - self.population) / 2
            self.best_pos = self.population
            self.best_val = values
            self.bestneigh_pos = self.population
            self.bestneigh_val = values
            self.initialized = True

        # update best positions
        better_result = values < self.best_val
        self.best_val[better_result] = values[better_result]
        self.best_pos[better_result] = self.population[better_result]
        # create constants
        K = 3 # TODO as parameter
        w = 1 / (2 * np.log(2))  # TODO as parameter
        c = 0.5 + np.log(2)  # TODO as parameter
        # inform other particles
        to_inform = np.random.choice(len(self.population), [K, len(self.population)])
        for neighbor in to_inform:
            to_update = values < self.bestneigh_val[neighbor]
            self.bestneigh_val[neighbor[to_update]] = values[to_update]
            self.bestneigh_pos[neighbor[to_update]] = self.population[to_update]

        # update velocity
        self.velocity = w * self.velocity + \
                        np.random.uniform(0, c, self.velocity.shape) * (self.best_pos - self.population) + \
                        np.random.uniform(0, c, self.velocity.shape) * (self.bestneigh_pos - self.population)
        # update positions
        self.population = self.population + self.velocity
        # confinement
        to_zero = np.logical_or(
            self.population < self.lowerbound[np.newaxis, :],
            self.population > self.upperbound[np.newaxis, :]
        )
        self.velocity[to_zero] = 0
        self.population = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.population))
        return self

    @staticmethod
    def initialize(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return Standard2006(lowerbound, upperbound, *shape)


class Standard2011:
    def __init__(self, lowerbound: np.ndarray, upperbound: np.ndarray, population, dimension):
        self.population = np.random.uniform(lowerbound, upperbound, [population, dimension])
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.initialized = False
        self.velocity = None
        self.best_pos = None
        self.best_val = None
        self.bestneigh_pos = None
        self.bestneigh_val = None

    def numpy(self):
        return self.population

    def execute(self, values: np.ndarray, function: cocoex.Problem):
        if not self.initialized:
            self.velocity = (np.random.uniform(self.lowerbound, self.upperbound, self.population.shape) - self.population) / 2
            self.best_pos = self.population
            self.best_val = values
            self.bestneigh_pos = self.population
            self.bestneigh_val = values
            self.initialized = True

        # update best positions
        better_result = values < self.best_val
        self.best_val[better_result] = values[better_result]
        self.best_pos[better_result] = self.population[better_result]
        # create constants
        K = 3 # TODO as parameter
        w = 1 / (2 * np.log(2))  # TODO as parameter
        c = 0.5 + np.log(2)  # TODO as parameter
        # inform other particles
        to_inform = np.random.choice(len(self.population), [K, len(self.population)])
        for neighbor in to_inform:
            to_update = values < self.bestneigh_val[neighbor]
            self.bestneigh_val[neighbor[to_update]] = values[to_update]
            self.bestneigh_pos[neighbor[to_update]] = self.population[to_update]

        # generate point on hypersphere
        p = self.population + np.random.uniform(0, c, self.velocity.shape) * (self.best_pos - self.population)
        l = self.population + np.random.uniform(0, c, self.velocity.shape) * (self.bestneigh_pos - self.population)
        G = (self.population + p + l) / 3
        x_ = np.random.randn(len(self.population), function.dimension)
        x_ /= np.linalg.norm(x_, axis=0)
        x_ += G

        # update velocity
        self.velocity = w * self.velocity + x_ - self.population
        # udpate population
        self.population = self.population + self.velocity
        # confinement
        to_zero = np.logical_or(
            self.population < self.lowerbound[np.newaxis, :],
            self.population > self.upperbound[np.newaxis, :]
        )
        self.velocity[to_zero] = -0.5 * self.velocity[to_zero]
        self.population = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.population))
        return self

    @staticmethod
    def initialize(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return Standard2011(lowerbound, upperbound, *shape)