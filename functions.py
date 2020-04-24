import cocoex
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

class _FunctionsWrapper:
    def __init__(self, suite):
        self._suite = suite  # type: cocoex.Suite
    def __enter__(self):
        return self._suite
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._suite.free()

def get_functions(function_indices = None, dimension = None, instance_indices='2'):
    function_indices = function_indices or []
    dimension = dimension or []
    function_indices_param = ",".join([str(n) for n in function_indices])
    dimension_param = ",".join([str(d) for d in dimension])
    if len(function_indices_param) > 0:
        function_indices_param = "function_indices:" + function_indices_param
    if len(dimension_param) > 0:
        dimension_param = "dimensions:" + dimension_param
    suite = cocoex.Suite("bbob", "", f"{dimension_param} {function_indices_param} instance_indices:{instance_indices}")
    return _FunctionsWrapper(suite)

def _getValues(function: cocoex.Problem):
    diff = 0.1
    x, y = np.arange(function.lower_bounds[0], function.upper_bounds[0] + diff, diff), np.arange(
        function.lower_bounds[1], function.upper_bounds[1] + diff, diff)
    X, Y = np.meshgrid(x, y)
    datapoints = np.stack([X.flatten(), Y.flatten()], axis=1)
    return np.apply_along_axis(function, axis=1, arr=datapoints), x, y, X, Y

def plot_function_flat(function: cocoex.Problem, figsize=(12,8)):
    values, x, y, X, Y = _getValues(function)
    plt.figure(figsize=figsize)
    plt.contourf(x, y, values.reshape([len(x), len(y)]), cmap='cool', levels=255)
    plt.title(function.name)
    plt.show()

def plot_function_3d(function: cocoex.Problem, figsize=(12,8)):
    values, x, y, X, Y = _getValues(function)
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, values.reshape([len(x), len(y)]), cmap='cool')
    plt.title(function.name)
    plt.show()