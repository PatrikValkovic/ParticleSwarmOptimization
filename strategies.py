import numpy as np
import cocoex

def random_walk(population: np.ndarray, values: np.ndarray, function: cocoex.Problem):
    return population + np.random.normal(0, 0.1, population.shape)


class RandomWalkObject:
    def __init__(self, lowerbound: np.ndarray, upperbound: np.ndarray, members: int, dimensions: int):
        self.population = np.random.uniform(lowerbound, upperbound, [members, dimensions])
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def numpy(self):
        return self.population

    def execute(self, values: np.ndarray, function: cocoex.Problem):
        self.population = random_walk(self.numpy(), values, function)
        self.population = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.population))
        return self

    @staticmethod
    def initialize(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return RandomWalkObject(lowerbound, upperbound, shape[0], shape[1])