"""
Genetic algorithm for M1-hard monitors.  Complements gradient descent by exploring
a broader region of parameter space and providing independent verification
of f(n, k).  The best individual is optionally refined by a short gradient run.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from src.core.loss import evaluate
from src.monitors.m1_unconstrained import M1HardMonitor
from src.optim.gradient import train_one


@dataclass
class Individual:
    W_H: np.ndarray  # (k, k) rows sum to 1
    W_T: np.ndarray
    omega: np.ndarray  # (k,)
    fitness: float = float("inf")


def _random_individual(k: int, rng: np.random.Generator, dirichlet_alpha: float = 1.0) -> Individual:
    W_H = rng.dirichlet(dirichlet_alpha * np.ones(k), size=k)
    W_T = rng.dirichlet(dirichlet_alpha * np.ones(k), size=k)
    omega = rng.uniform(-3.0, 3.0, size=k)
    return Individual(W_H=W_H, W_T=W_T, omega=omega)


def _to_monitor(ind: Individual, dtype=torch.float64) -> M1HardMonitor:
    k = ind.W_H.shape[0]
    m = M1HardMonitor(k=k, dtype=dtype)
    eps = 1e-6
    with torch.no_grad():
        m.W_H.data.copy_(torch.tensor(np.log(np.clip(ind.W_H, eps, None)), dtype=dtype))
        m.W_T.data.copy_(torch.tensor(np.log(np.clip(ind.W_T, eps, None)), dtype=dtype))
        m.omega.data.copy_(torch.tensor(ind.omega, dtype=dtype))
    return m


def _fitness(ind: Individual, n: int) -> float:
    return evaluate(_to_monitor(ind), n)


def _tournament(pop: List[Individual], rng: np.random.Generator, k_sel: int = 3) -> Individual:
    idxs = rng.integers(0, len(pop), size=k_sel)
    return min((pop[i] for i in idxs), key=lambda x: x.fitness)


def _crossover(a: Individual, b: Individual, rng: np.random.Generator) -> Individual:
    # Row-wise mixing, then re-normalise (averaging two probability rows keeps them stochastic).
    mask_H = rng.random(size=a.W_H.shape[0]) < 0.5
    mask_T = rng.random(size=a.W_T.shape[0]) < 0.5
    W_H = np.where(mask_H[:, None], a.W_H, b.W_H)
    W_T = np.where(mask_T[:, None], a.W_T, b.W_T)
    omega = np.where(rng.random(size=a.omega.shape) < 0.5, a.omega, b.omega)
    return Individual(W_H=W_H, W_T=W_T, omega=omega)


def _mutate(ind: Individual, rng: np.random.Generator, rate: float = 0.1, sigma: float = 0.1) -> None:
    if rng.random() < rate:
        ind.W_H = np.clip(ind.W_H + rng.normal(0, sigma, size=ind.W_H.shape), 1e-6, None)
        ind.W_H /= ind.W_H.sum(axis=1, keepdims=True)
    if rng.random() < rate:
        ind.W_T = np.clip(ind.W_T + rng.normal(0, sigma, size=ind.W_T.shape), 1e-6, None)
        ind.W_T /= ind.W_T.sum(axis=1, keepdims=True)
    if rng.random() < rate:
        ind.omega = ind.omega + rng.normal(0, sigma, size=ind.omega.shape)


def run_ga(
    k: int,
    n: int,
    population_size: int = 200,
    generations: int = 500,
    mutation_rate: float = 0.1,
    gradient_refine_steps: int = 100,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[M1HardMonitor, float]:
    rng = np.random.default_rng(seed)
    pop = [_random_individual(k, rng) for _ in range(population_size)]
    for ind in pop:
        ind.fitness = _fitness(ind, n)

    for gen in range(generations):
        pop.sort(key=lambda x: x.fitness)
        new_pop = pop[: population_size // 10]  # elitism
        while len(new_pop) < population_size:
            a = _tournament(pop, rng)
            b = _tournament(pop, rng)
            child = _crossover(a, b, rng)
            _mutate(child, rng, rate=mutation_rate)
            child.fitness = _fitness(child, n)
            new_pop.append(child)
        pop = new_pop
        if verbose and gen % max(1, generations // 20) == 0:
            print(f"  gen {gen:>4d}  best = {pop[0].fitness:.6f}")

    best = pop[0]
    monitor = _to_monitor(best)
    if gradient_refine_steps > 0:
        res = train_one(monitor, n, lr=1e-3, steps=gradient_refine_steps)
        return monitor, res.final_loss
    return monitor, best.fitness
