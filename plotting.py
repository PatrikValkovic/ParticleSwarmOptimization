from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import cocoex
import io
import os
from PIL import Image
from progressbar import progressbar

def _getValues(function: cocoex.Problem):
    diff = 0.1
    x, y = np.arange(function.lower_bounds[0], function.upper_bounds[0] + diff, diff), np.arange(
        function.lower_bounds[1], function.upper_bounds[1] + diff, diff)
    X, Y = np.meshgrid(x, y)
    datapoints = np.stack([X.flatten(), Y.flatten()], axis=1)
    return np.apply_along_axis(function, axis=1, arr=datapoints), x, y, X, Y

def _plot_contours(function: cocoex.Problem) -> None:
    """
    Plot function as matplotlib.contourf graph.
    :param function: Function to plot.
    """
    values, x, y, X, Y = _getValues(function)
    plt.imshow(values.reshape([len(x), len(y)]), cmap='cool', extent=(-5, 5, -5, 5), interpolation='hermite', origin='lower')
    #plt.contourf(x, y, values.reshape([len(x), len(y)]), cmap='cool', levels=255)


def plot_function_flat(function: cocoex.Problem, figsize=(12,12)):
    plt.figure(figsize=figsize)
    _plot_contours(function)
    plt.title(function.name)
    plt.show()

def plot_function_3d(function: cocoex.Problem, figsize=(12,8)):
    values, x, y, X, Y = _getValues(function)
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, values.reshape([len(x), len(y)]), cmap='cool')
    plt.title(function.name)
    plt.show()

def plot_population(function: cocoex.Problem, population, figsize=(12,12), title=None, color='r') -> None:
    """
    Plot population over the function.
    :param function: Function over which to plot (uses matplotlib.contourf).
    :param population: Population in shape (population_size, 2).
    :param figsize: Figure size
    :param title: Title of the figure, by default uses `function.name`.
    :param color: Color of the population.
    """
    plt.figure(figsize=figsize)
    _plot_contours(function)
    plt.scatter(population[:,0], population[:,1], c=color)
    plt.title(title or function.name)
    plt.show()


def plot_movement_of_individual(function: cocoex.Problem, member, figsize=(12, 12), title=None, color='r', line_alpha=0.1) -> None:
    """
    Plot how individual moved over the generations.
    :param function: Function over which to plot (uses matplotlib.contourf).
    :param member: Member of the population in shape (generations, 2).
    :param figsize: Figure size.
    :param title: Title of the figure, by default uses `function.name`.
    :param color: Color of the individual dots.
    :param line_alpha: Alpha of the lines connecting member's positions.
    """
    alpha_channel = np.linspace(0.01, 1.0, member.shape[0]).reshape(-1,1)
    base_color = pltcolors.to_rgb(color)
    final_color = np.concatenate([np.array([base_color]).repeat(member.shape[0], axis=0), alpha_channel], axis=1)

    plt.figure(figsize=figsize)
    _plot_contours(function)
    plt.plot(member[:,0], member[:,1], c=color, alpha=line_alpha)
    plt.scatter(member[:,0], member[:,1], color=final_color)
    plt.title(title or function.name)
    plt.show()

def animate_movement(function: cocoex.Problem, populations, show_progress=False, figsize=(12,12), color='r', title=None) -> bytearray:
    """
    Transform population movement over the function.
    :param function: Function over which to plot (uses matplotlib.contourf).
    :param populations: Population in the shape (generations, population_size, 2).
    :param figsize: Figure size.
    :param color: Color of the dots representing members.
    :param title: Title of the plot, by default uses `function.name`.
    :return: Bytearray representing gif image.
    """
    frames = []
    buffers = []

    iteration = zip(range(populations.shape[0]), populations)
    if show_progress:
        iteration = progressbar(iteration, max_value=populations.shape[0])

    values, x, y, X, Y = _getValues(function)

    for gen, population in iteration:
        fig = plt.figure(figsize=figsize)
        plt.imshow(values.reshape([len(x), len(y)]), cmap='cool', extent=(-5,5,-5,5), interpolation='hermite', origin='lower')
        plt.scatter(population[:, 0], population[:, 1], c=color)
        plt.title(title or function.name)
        buffers.append(io.BytesIO())
        plt.savefig(buffers[-1], format='jpg')
        buffers[-1].seek(0)
        frames.append(Image.open(buffers[-1]))
        plt.close(fig)
    frames[0].save("tmp.gif", save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
    for frame, buffer in zip(frames, buffers):
        buffer.close()
    with open("tmp.gif", 'rb') as f:
        content = f.read()
    os.remove("tmp.gif")
    return content


def plot_graph(fitnesses: np.array, plotfn: Callable, **plot_params) -> None:
    """
    Plot line graph over the population. This function doesn't call `show` method.
    :param fitnesses: Population fitnesses in the shape (generations, population_size)
    :param plotfn: Aggregation function that should transform `fitnesses` into 1D array.
    :param title: Title of the graph, if should be used.
    :param plot_params: Params to be passed to the `matplotlib.plot` call.
    """
    to_plot = plotfn(fitnesses)
    x_coords = np.arange(1, len(to_plot) + 1, dtype=int)
    plt.plot(x_coords, to_plot, **plot_params)

def plot_aggregated(fitnesses: np.array, plotfn: Callable, aggregationfn: Callable, **plot_params) -> None:
    """
    Plot line graph over the aggregated data. This function doesn't call `show` method.
    :param fitnesses: Population fitnesses in the shape (repeats, generations, population_size)
    :param plotfn: Aggregation function that should transform `fitnesses` into 1D array named `tmp`.
    :param aggregationfn: Aggregation function that should transform array of shape (repeats, len(tmp)) into 1D array.
    :param plot_params: Params to be passed to the `matplotlib.plot` call.
    """
    plots = np.stack([
        plotfn(fitness) for fitness in fitnesses
    ])
    aggregated = aggregationfn(plots)
    x_coords = np.arange(1, len(aggregated) + 1, dtype=int)
    plt.plot(x_coords, aggregated, **plot_params)



def popfn_quantile(q):
    return lambda v: np.quantile(v, q=q, axis=1)

def popfn_median():
    return popfn_quantile(0.5)

def popfn_90():
    return popfn_quantile(0.9)

def popfn_95():
    return popfn_quantile(0.95)

def popfn_mean():
    return lambda v: np.mean(v, axis=1)

def popfn_max():
    return lambda v: np.max(v, axis=1)

def popfn_min():
    return lambda v: np.min(v, axis=1)


def aggfn_quantile(q):
    return lambda v: np.quantile(v, q=q, axis=0)

def aggfn_median():
    return aggfn_quantile(0.5)

def aggfn_90():
    return aggfn_quantile(0.9)

def aggfn_95():
    return aggfn_quantile(0.95)

def aggfn_mean():
    return lambda v: np.mean(v, axis=0)

def aggfn_max():
    return lambda v: np.max(v, axis=0)

def aggfn_min():
    return lambda v: np.min(v, axis=0)