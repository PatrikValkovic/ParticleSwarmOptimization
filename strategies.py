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
    return particles - gradients * computation_step


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