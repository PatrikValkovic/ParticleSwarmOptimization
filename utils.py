import cocoex
import numpy as np
from execution import evaluate


def compute_gradient(
        function: cocoex.Problem,
        values: np.ndarray,
        gradient_step: float = 0.001
) -> np.ndarray:
    gradients = []
    for dimension in range(function.dimension):
        population_minus = values.copy()
        population_minus[:, dimension] = population_minus[:, dimension] - gradient_step
        evaluation_minus = evaluate(function, population_minus)
        population_plus = values.copy()
        population_plus[:, dimension] = population_plus[:, dimension] + gradient_step
        evaluation_plus = evaluate(function, population_plus)
        gradients.append((evaluation_plus - evaluation_minus) / (2 * gradient_step))
    return np.stack(gradients, axis=-1)