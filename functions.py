import cocoex
import matplotlib.pyplot as plt
import numpy as np

class _FunctionsWrapper:
    def __init__(self, suite):
        self._suite = suite  # type: cocoex.Suite
    def __enter__(self):
        return self._suite
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._suite.free()

def get_suite(function_indices = None, dimension = None, instance_indices='2'):
    function_indices = function_indices or []
    dimension = dimension or []
    function_indices_param = ",".join([str(n) for n in function_indices])
    dimension_param = ",".join([str(d) for d in dimension])
    if len(function_indices_param) > 0:
        function_indices_param = "function_indices:" + function_indices_param
    if len(dimension_param) > 0:
        dimension_param = "dimensions:" + dimension_param
    return cocoex.Suite("bbob", "", f"{dimension_param} {function_indices_param} instance_indices:{instance_indices}")

def get_suite_wrapper(function_indices = None, dimension = None, instance_indices='2'):
    return _FunctionsWrapper(get_suite(function_indices, dimension, instance_indices))
