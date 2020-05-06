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


def random_walk(particles: np.ndarray, *vars, stepsize: float = 0.2) -> np.ndarray:
    particles = particles  + np.random.normal(0, stepsize, particles.shape)
    return np.maximum(-5, np.minimum(5, particles))


class RandomVelocity:
    def __init__(self, lowerbound, upperbound, populationsize, dimension, velocity_lowerbound, velocity_upperbound):
        self.particles = np.random.uniform(lowerbound, upperbound, (populationsize, dimension))
        self.velocities = np.random.uniform(velocity_lowerbound, velocity_upperbound, (populationsize, dimension))
    def numpy(self):
        return self.particles
    def execute(self, *vars, w: float = 1.0, stepsize: float = 0.05):
        self.velocities = w * self.velocities + np.random.normal(0, stepsize, self.velocities.shape)
        self.particles = np.maximum(-5, np.minimum(5, self.particles + self.velocities))
        return self
    @staticmethod
    def init(lowerbound, upperbound, shape, velocity_lowerbound=0.1, velocity_upperbound=0.1):
        return RandomVelocity(lowerbound, upperbound, shape[0], shape[1], velocity_lowerbound, velocity_upperbound)


def differential_evolution(population: np.ndarray, fitnesses: np.ndarray, function:cocoex.Problem,
                           parents_fraction: float = 0.6,
                           F: float = 0.8,
                           CR: float = 0.4,
                           ) -> np.ndarray:
    # pickup parents
    num_parents = int(len(population) * parents_fraction)
    parents_tournament_indices = np.random.randint(0, len(population), [2, num_parents])
    comparison = fitnesses[parents_tournament_indices[0]] < fitnesses[parents_tournament_indices[1]]
    better = np.concatenate([
        parents_tournament_indices[0, comparison],
        parents_tournament_indices[1, np.logical_not(comparison)]
    ])
    parents = population[better]

    # create children
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


class FollowBest:
    def __init__(self, lowerbound, upperbound, populationsize, dimension, velocity_lowerbound, velocity_upperbound):
        self.particles = np.random.uniform(lowerbound, upperbound, (populationsize, dimension))
        self.velocities = np.random.uniform(velocity_lowerbound, velocity_upperbound, (populationsize, dimension))
    def numpy(self):
        return self.particles
    def execute(self, values, *vars, w = 1 / (2 * np.log(2)), c = 0.5 + np.log(2), stepsize = 0.05):
        best = np.argmin(values)
        best_individual = self.particles[best]
        random_walk = np.random.normal(0, stepsize, self.velocities.shape)
        follow_best = np.random.uniform(0, c, len(self.particles))[:,np.newaxis] * (best_individual - self.particles)
        self.velocities = w * self.velocities + random_walk + follow_best
        self.particles = np.maximum(-5, np.minimum(5, self.particles + self.velocities))
        return self
    @staticmethod
    def init(lowerbound, upperbound, shape, velocity_lowerbound=0.1, velocity_upperbound=0.1):
        return FollowBest(lowerbound, upperbound, shape[0], shape[1], velocity_lowerbound, velocity_upperbound)


class RingTopology:
    def __init__(self, lowerbound, upperbound, populationsize, dimension, velocity_lowerbound, velocity_upperbound):
        self.particles = np.random.uniform(lowerbound, upperbound, (populationsize, dimension))
        self.velocities = np.random.uniform(velocity_lowerbound, velocity_upperbound, (populationsize, dimension))
    def numpy(self):
        return self.particles
    def execute(self, values, *vars, w = 1 / (2 * np.log(2)), c = 0.5 + np.log(2), stepsize = 0.05, K=3):
        N = len(self.particles)
        neighbors = np.linspace(list(range(1,1+K)), list(range(N,N+K)), N, dtype=int) % N
        best_neighbor_indices = (np.arange(1,N+1) + np.argmin(values[neighbors], axis=1)) % N
        best_neighbor = self.particles[best_neighbor_indices]
        random_walk = np.random.normal(0, stepsize, self.velocities.shape)
        follow_best = np.random.uniform(0, c, len(self.particles))[:,np.newaxis] * (best_neighbor - self.particles)
        self.velocities = w * self.velocities + random_walk + follow_best
        self.particles = np.maximum(-5, np.minimum(5, self.particles + self.velocities))
        return self
    @staticmethod
    def init(lowerbound, upperbound, shape, velocity_lowerbound=0.1, velocity_upperbound=0.1):
        return RingTopology(lowerbound, upperbound, shape[0], shape[1], velocity_lowerbound, velocity_upperbound)


class RandomTopology:
    def __init__(self, lowerbound, upperbound, populationsize, dimension, velocity_lowerbound, velocity_upperbound):
        self.particles = np.random.uniform(lowerbound, upperbound, (populationsize, dimension))
        self.velocities = np.random.uniform(velocity_lowerbound, velocity_upperbound, (populationsize, dimension))
    def numpy(self):
        return self.particles
    def execute(self, values, *vars, w = 1 / (2 * np.log(2)), c = 0.5 + np.log(2), stepsize = 0.05, K=3):
        N = len(self.particles)
        neighbors = np.random.randint(0, N, (N, K))
        best_neighbor_indices = neighbors[range(N), np.argmin(values[neighbors], axis=1)]
        best_neighbor = self.particles[best_neighbor_indices]
        random_walk = np.random.normal(0, stepsize, self.velocities.shape)
        follow_best = np.random.uniform(0, c, len(self.particles))[:,np.newaxis] * (best_neighbor - self.particles)
        self.velocities = w * self.velocities + follow_best + random_walk
        self.particles = np.maximum(-5, np.minimum(5, self.particles + self.velocities))
        return self
    @staticmethod
    def init(lowerbound, upperbound, shape, velocity_lowerbound=0.1, velocity_upperbound=0.1):
        return RandomTopology(lowerbound, upperbound, shape[0], shape[1], velocity_lowerbound, velocity_upperbound)


class NearestTopology:
    def __init__(self, lowerbound, upperbound, populationsize, dimension, velocity_lowerbound, velocity_upperbound):
        self.particles = np.random.uniform(lowerbound, upperbound, (populationsize, dimension))
        self.velocities = np.random.uniform(velocity_lowerbound, velocity_upperbound, (populationsize, dimension))
    def numpy(self):
        return self.particles
    def execute(self, values, *vars, w = 1 / (2 * np.log(2)), c = 0.5 + np.log(2), stepsize = 0.05, K=3):
        N = len(self.particles)
        neighbors = np.zeros((N,K), dtype=int)
        for particle_i in range(N):
            distances = np.sqrt(np.sum((self.particles[particle_i][np.newaxis, :] - self.particles) ** 2, axis=1))
            closest = np.argsort(distances)[1:K+1]
            neighbors[particle_i] = closest
        best_neighbor_indices = neighbors[range(N), np.argmin(values[neighbors], axis=1)]
        best_neighbor = self.particles[best_neighbor_indices]
        random_walk = np.random.normal(0, stepsize, self.velocities.shape)
        follow_best = np.random.uniform(0, c, len(self.particles))[:,np.newaxis] * (best_neighbor - self.particles)
        self.velocities = w * self.velocities + random_walk + follow_best
        self.particles = np.maximum(-5, np.minimum(5, self.particles + self.velocities))
        return self
    @staticmethod
    def init(lowerbound, upperbound, shape, velocity_lowerbound=0.1, velocity_upperbound=0.1):
        return RandomTopology(lowerbound, upperbound, shape[0], shape[1], velocity_lowerbound, velocity_upperbound)

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

    def execute(self, values: np.ndarray, function: cocoex.Problem,
                K: int = 3,
                w: float = 1 / (2 * np.log(2)),
                c: float = 0.5 + np.log(2)):
        if not self.initialized:
            self.velocity = (np.random.uniform(self.lowerbound, self.upperbound, self.population.shape) - self.population) / 2
            self.best_pos = self.population.copy()
            self.best_val = values.copy()
            self.bestneigh_pos = self.population.copy()
            self.bestneigh_val = values.copy()
            self.initialized = True

        # update best positions
        better_result = values < self.best_val
        self.best_val[better_result] = values[better_result]
        self.best_pos[better_result] = self.population[better_result]

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
    def init(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return Standard2006(lowerbound, upperbound, *shape)


class Standard2011:
    def __init__(self, lowerbound: np.ndarray, upperbound: np.ndarray, population, dimension):
        self.particles = np.random.uniform(lowerbound, upperbound, [population, dimension])
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.initialized = False
        self.velocity = None
        self.best_pos = None
        self.best_val = None
        self.bestneigh_pos = None
        self.bestneigh_val = None

    def numpy(self):
        return self.particles

    def _sample_from_sphere(self, center, radius, shape):
        Y = np.random.normal(0, 1, shape)
        u = np.random.random(shape[0])
        r = radius * np.power(u, 1. / shape[1])
        norm = np.linalg.norm(Y, axis=1)
        x = (r/norm)[:, np.newaxis] * Y
        assert (np.linalg.norm(x, axis=1) <= radius).all()
        return x+center

    def execute(self, values: np.ndarray, function: cocoex.Problem,
                K: int = 3,
                w: float = 1 / (2 * np.log(2)),
                c: float = 0.5 + np.log(2)):
        if not self.initialized:
            self.velocity = (np.random.uniform(self.lowerbound, self.upperbound, self.particles.shape) - self.particles) / 2
            self.best_pos = self.particles.copy()
            self.best_val = values.copy()
            self.bestneigh_pos = self.particles.copy()
            self.bestneigh_val = values.copy()
            self.initialized = True

        # update best positions
        better_result = values < self.best_val
        self.best_val[better_result] = values[better_result]
        self.best_pos[better_result] = self.particles[better_result]

        # inform other particles
        to_inform = np.random.choice(len(self.particles), [K, len(self.particles)])
        for neighbor in to_inform:
            to_update = values < self.bestneigh_val[neighbor]
            self.bestneigh_val[neighbor[to_update]] = values[to_update]
            self.bestneigh_pos[neighbor[to_update]] = self.particles[to_update]

        # generate point in hypersphere
        bg = self.particles + np.random.uniform(0, c, self.particles.shape) * (self.bestneigh_pos - self.particles)
        bl = self.particles + np.random.uniform(0, c, self.particles.shape) * (self.best_pos - self.particles)
        G = (self.particles + bl + bg) / 3
        radius = np.linalg.norm(G - self.particles, axis=1)
        x_ = self._sample_from_sphere(G, radius, self.velocity.shape)

        # update velocity
        self.velocity = w * self.velocity + x_ - self.particles
        # udpate population
        self.particles = self.particles + self.velocity
        # confinement
        to_zero = np.logical_or(
            self.particles < self.lowerbound[np.newaxis, :],
            self.particles > self.upperbound[np.newaxis, :]
        )
        self.velocity[to_zero] = -0.5 * self.velocity[to_zero]
        self.particles = np.maximum(self.lowerbound, np.minimum(self.upperbound, self.particles))
        return self

    @staticmethod
    def init(lowerbound: np.ndarray, upperbound: np.ndarray, shape:tuple):
        return Standard2011(lowerbound, upperbound, *shape)