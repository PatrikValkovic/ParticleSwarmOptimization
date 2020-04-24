import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from functions import _getValues
import cocoex
import io
import os
from PIL import Image
from IPython import display
import base64

def _plot_contours(function: cocoex.Problem):
    values, x, y, X, Y = _getValues(function)
    plt.contourf(x, y, values.reshape([len(x), len(y)]), cmap='cool', levels=255)


def plot_population(function: cocoex.Problem, population, figsize=(12,8), title=None, color='r'):
    plt.figure(figsize=figsize)
    _plot_contours(function)
    plt.scatter(population[:,0], population[:,1], c=color)
    plt.title(title or function.name)
    plt.show()


def plot_movement_of_individual(function: cocoex.Problem, member, figsize=(12, 8), title=None, color='r', line_alpha=0.1):
    alpha_channel = np.linspace(0.01, 1.0, member.shape[0]).reshape(-1,1)
    base_color = pltcolors.to_rgb(color)
    final_color = np.concatenate([np.array([base_color]).repeat(member.shape[0], axis=0), alpha_channel], axis=1)

    plt.figure(figsize=figsize)
    _plot_contours(function)
    plt.plot(member[:,0], member[:,1], c=color, alpha=line_alpha)
    plt.scatter(member[:,0], member[:,1], color=final_color)
    plt.title(title or function.name)
    plt.show()

def animate_movement(function: cocoex.Problem, populations, figsize=(12,8), color='r', title=None):
    frames = []
    buffers = []
    for gen, population in zip(range(populations.shape[0]), populations):
        fig = plt.figure(figsize=figsize)
        _plot_contours(function)
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
