import numpy as np

def random_walk(population, values):
    return population + np.random.normal(0, 0.1, population.shape)