import numpy as np

def random_walk(population, values):
    return population + np.random.normal(0, 0.1, population.shape)


class RandomWalkObject:
    def __init__(self, lowerbound, upperbound, members, dimensions):
        self.population = np.random.uniform(lowerbound, upperbound, [members, dimensions])
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def numpy(self):
        return self.population

    def execute(self, values):
        self.population = random_walk(self.numpy(), values)
        self.population = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.population))
        return self

    @staticmethod
    def initialize(lowerbound, upperbound, shape):
        return RandomWalkObject(lowerbound, upperbound, shape[0], shape[1])